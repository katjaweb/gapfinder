import json
from typing import Any
from IPython.display import display, Markdown


class YouTubeKnowledgeExtractor:
    """Helper class for extracting and formatting YouTube transcripts."""
    
    @staticmethod
    def get_video_id(url: str) -> str:
        """Extract video ID from YouTube URL"""
        if "v=" in url:
            return url.split("v=")[1][:11]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1][:11]
        return url

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS"""
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}:{minutes:02}:{secs:02}"
        else:
            return f"{minutes}:{secs:02}"

    @staticmethod
    def get_transcript(video_id: str, ytt_api: Any):
        """Download transcript"""
        transcript = ytt_api.fetch(video_id)
        return transcript.snippets

    @staticmethod
    def make_subtitles(transcript) -> str:
        """Convert transcript entries into text with timestamps."""
        lines = []
        for entry in transcript:
            ts = YouTubeKnowledgeExtractor.format_timestamp(entry.start)
            text = entry.text.replace('\n', ' ')
            lines.append(f"{ts} {text}")
        return '\n'.join(lines)


class GapFinderAgentTools:
    """
    This class holds the state (e.g. the search index) and all tools.
    """
    
    def __init__(self, client: Any, model: str, ytt_api: Any, chunk_func: Any, index_cls: Any):
        self.client = client
        self.model = model
        self.ytt_api = ytt_api
        self.chunk_func = chunk_func
        self.index_cls = index_cls
        
        self.current_index = None

    def extract_concepts(self, transcript_text: str) -> str:
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
        print('extracting concepts from transcript text')
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        print('extracting concepts complete')
        return response.choices[0].message.content

    def process_video_transcript(self, video_url: str) -> str:
        """
        Download a YouTube transcript, split it into chunks,
        and build the searchable transcript index.

        IMPORTANT:
        This tool DOES NOT extract concepts.
        After calling this tool, the agent MUST call
        `extract_concepts`.
        """

        print(f"Processing video URL: {video_url}")

        video_id = YouTubeKnowledgeExtractor.get_video_id(video_url)

        try:
            transcript = YouTubeKnowledgeExtractor.get_transcript(
                video_id,
                self.ytt_api
            )

        except Exception as e:
            return f"Error downloading transcript: {str(e)}"

        subtitles = YouTubeKnowledgeExtractor.make_subtitles(transcript)

        transcript_text = " ".join(
            [segment.text for segment in transcript]
        )

        print("Building search index...")

        chunks = self.chunk_func(
            [{'content': subtitles}],
            size=3000,
            step=500
        )

        self.current_index = self.index_cls(
            text_fields=['content']
        )

        self.current_index.fit(chunks)

        print("Video processing complete.")

        return transcript_text

    def search_video_transcript(self, search_query: str) -> str:
        """
        Performs a lexical search over the video's transcript to find specific explanations.
        """
        if self.current_index is None:
            return "Error: No video has been processed yet. Please ask the user to provide a YouTube URL first."
        
        print(f"Searching transcript for: '{search_query}'")
        results = self.current_index.search(search_query, num_results=5)
        return json.dumps(results, indent=2)

    def evaluate_user_answer(self, question: str, user_answer: str, reference_context: str) -> str:
        """
        Uses the strict GapFinder rubric to grade a user's answer.
        """
        print(f"Evaluating user answer...")
        prompt = f"""
        Evaluate the user's answers and provide a markdown-formatted response with:
        - What they understood well
        - What they misunderstood or missed
        - What to revisit
        
        <QUESTION>
        {question}
        </QUESTION>

        
        <CONTEXT>
        {reference_context}
        </CONTEXT>
        
        <USER_ANSWERS>
        {user_answer}
        </USER_ANSWERS>
        """
        print(f"question: {question}")
        print(f"reference context: {reference_context}")
        print(f"user answer: {user_answer}")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert tutor grading a student's answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content