import argparse
import os
import json
from pathlib import Path
from typing import Any, List, Dict
import logging

from minsearch import Index
from youtube_transcript_api import YouTubeTranscriptApi

from gitsource import chunk_documents
from openai import OpenAI
import yt_dlp
import dotenv
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoMetadataService:
    """Extract metadata from YouTube videos."""
    
    
    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract video ID from YouTube URL"""
        if "v=" in url:
            return url.split("v=")[1][:11]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1][:11]
        return url

    @staticmethod
    def fetch_metadata(video_url: str) -> dict:
        ydl_opts = {"quiet": True, "skip_download": True}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        return {
            "video_id": info.get("id"),
            "title": info.get("title"),
        }


class TranscriptService:
    """Handles transcript fetching and formatting."""

    def __init__(self, transcript_api: Any):
        self.api = transcript_api

    def fetch_transcript(self, video_id: str):
        """Download transcript"""
        transcript = self.api.fetch(video_id)
        return transcript


    def transcript_to_text(self, transcript) -> str:
        text = " ".join(
            entry["text"] if isinstance(entry, dict) else entry.text
            for entry in transcript
        )
        return text
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS"""
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}:{minutes:02}:{secs:02}"
        else:
            return f"{minutes}:{secs:02}"
    
    
    def make_subtitles(self, transcript) -> str:
        """Convert transcript entries into text with timestamps."""
        lines = []
        for entry in transcript.snippets:
            ts = self.format_timestamp(entry.start)
            text = entry.text.replace('\n', ' ')
            lines.append(f"{ts} {text}")
        return '\n'.join(lines)


class StorageService:
    """Handles JSON persistence for transcripts and chunks."""

    def __init__(self, base_path: str = None):
        if base_path is None:
            project_root = Path(__file__).resolve().parent.parent
            base_path = project_root / "data"

        self.base_path = str(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    def transcript_file_path(self) -> str:
        return os.path.join(self.base_path, "transcripts.json")

    # ---------- entries ----------

    def load_entries(self, file_path: str) -> List[Dict]:
        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def save_entries(self, file_path: str, data: List[Dict]) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def append_entry(self, file_path: str, entry: Dict) -> None:
        data = self.load_entries(file_path)

        entry_video_id = next(iter(entry), None)
        if entry_video_id is None:
            return

        for existing in data:
            if entry_video_id in existing:
                existing[entry_video_id].update(entry[entry_video_id])
                self.save_entries(file_path, data)
                return
            if existing.get("video_id") == entry_video_id:
                existing.update(entry)
                self.save_entries(file_path, data)
                return

        data.append(entry)
        self.save_entries(file_path, data)

    # ---------- chunks ----------

    def chunk_file_path(self) -> str:
        return os.path.join(self.base_path, "yt_chunks.json")

    def load_chunks(self) -> List[Dict]:
        return self.load_entries(self.chunk_file_path())

    def save_chunks(self, chunks: List[Dict]) -> None:
        self.save_entries(self.chunk_file_path(), chunks)


class ChunkService:
    """Handles document chunking logic."""

    def __init__(self, storage):
        self.storage = storage

    def store_chunks(self, video_id, title, subtitles, chunk_fn):
        chunks = self.storage.load_chunks()

        if any(c.get("video_id") == video_id for c in chunks):
            logger.info(f"Chunks for video_id {video_id} already exist. Skipping chunking.")
            return chunks

        raw_docs = [{
            "video_id": video_id,
            "title": title,
            "content": subtitles
        }]

        new_chunks = chunk_fn(raw_docs, size=3000, step=500)

        for i, c in enumerate(new_chunks):
            c["video_id"] = video_id
            c["chunk_index"] = i

        chunks.extend(new_chunks)
        self.storage.save_chunks(chunks)

        return chunks


class YouTubePipeline:
    """Main orchestration pipeline."""

    def __init__(
        self,
        metadata_service=None,
        transcript_service=None,
        storage_service=None,
        chunk_service=None,
        chunk_documents_fn=None,
        openai_client=None
    ):
        self.metadata = metadata_service
        self.transcripts = transcript_service
        self.storage = storage_service
        self.chunk_service = chunk_service
        self.chunk_documents_fn = chunk_documents_fn
        self.client = openai_client or OpenAI()
        self.entries = {}

    # ----------------------------------------
    # Helpers
    # ----------------------------------------

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        seconds = int(seconds)

        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours:
            return f"{hours}:{minutes:02}:{secs:02}"

        return f"{minutes}:{secs:02}"

    # ----------------------------------------
    # Main Flow
    # ----------------------------------------

    def extract_concepts(self, transcript_text: str, model: str = "gpt-4o-mini") -> str:
        """
        Extract the most important educational concepts from a transcript.

        IMPORTANT:
        This tool should only be called AFTER the transcript
        has already been processed and indexed.
        """

        prompt = f"""
        You are an expert educator.

        Extract the 3-5 most important concepts from the following
        educational video transcript.

        For each concept provide:
        - a short title
        - a brief explanation

        Return the result as a clean bulleted list.

        Transcript:
        {transcript_text}
        """
        logger.info('extracting concepts from transcript text')
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        logger.info('extracting concepts complete')
        return response.choices[0].message.content

    def create_rag_index(self, chunks: List[Dict] = None) -> Index:
        """
        Create and fit a RAG index for lexical search over chunks.
        
        Args:
            chunks: List of chunk dictionaries. If None, loads from storage.
            
        Returns:
            Index: Fitted minsearch Index ready for retrieval.
        """
        if chunks is None:
            chunks = self.storage.load_chunks()
        
        index = Index(
            text_fields=["content", "title"],
            keyword_fields=["video_id"],
        )
        
        logger.info(f"Fitting RAG index with {len(chunks)} chunks")
        index.fit(chunks)
        
        return index

    def process_video(
        self,
        url: str,
        generate_summary: bool = True,
        generate_chunks: bool = True,
    ) -> dict:

        # ----------------------------------------
        # 1. Metadata
        # ----------------------------------------

        meta = self.metadata.fetch_metadata(url)

        video_id = meta["video_id"]
        title = meta["title"]
        
        chunks = self.storage.load_chunks()
        if any(c.get("video_id") == video_id for c in chunks):
            logger.info(f"Chunks for video_id {video_id} already exist. Skipping chunking.")

        logger.info(f"Processing: {title}")

        # ----------------------------------------
        # 2. Transcript
        # ----------------------------------------

        transcript = self.transcripts.fetch_transcript(video_id)

        transcript_text = self.transcripts.transcript_to_text(transcript)

        subtitles = self.transcripts.make_subtitles(transcript)

        # ----------------------------------------
        # 3. Base Entry
        # ----------------------------------------
        self.entries[video_id] = {
            "title": title,
            "transcript_text": transcript_text,
        }

        # ----------------------------------------
        # 4. Optional Summary
        # ----------------------------------------

        if generate_summary:
            try:
                summary = self.extract_concepts(
                    transcript_text
                )
                self.entries[video_id]["summary"] = summary

            except Exception as e:
                logger.error(f"Summary generation failed: {e}")

        # ----------------------------------------
        # 5. Save Transcript Entry
        # ----------------------------------------

        transcripts_path = self.storage.transcript_file_path()
        self.storage.append_entry(
            transcripts_path,
            self.entries
        )

        # ----------------------------------------
        # 6. Optional Chunking
        # ----------------------------------------

        if (
            generate_chunks
            and self.chunk_service
            and self.chunk_documents_fn
        ):

            chunk_source = (subtitles)

            self.chunk_service.store_chunks(
                video_id=video_id,
                title=title,
                subtitles=chunk_source,
                chunk_fn=self.chunk_documents_fn
            )

        logger.info(f"Finished processing: {video_id}")

        return chunks
    

if __name__ == "__main__":

    ytt_api = YouTubeTranscriptApi()
    storage_service = StorageService()

    parser = argparse.ArgumentParser(
        description="Build a YouTube RAG index from a video URL."
    )
    parser.add_argument(
        "url",
        help="YouTube video URL or video ID to process.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summary extraction and only create chunks and index.",
    )
    args = parser.parse_args()

    pipeline = YouTubePipeline(
        metadata_service=VideoMetadataService(),
        transcript_service=TranscriptService(ytt_api),
        storage_service=storage_service,
        chunk_service=ChunkService(storage_service),
        chunk_documents_fn=chunk_documents,
    )

    pipeline.process_video(
        url=args.url,
        generate_summary=not args.no_summary,
        generate_chunks=True,
    )

    index = pipeline.create_rag_index()
    logger.info(f"RAG index created with {len(index.docs)} chunks")
