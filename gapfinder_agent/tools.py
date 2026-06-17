import json
from typing import Any
from IPython.display import display, Markdown


class GapFinderAgentTools:
    """
    This class holds the state (e.g. the search index) and all tools.
    """
    
    def __init__(self, client: Any, model: str, index_cls: Any):
        self.client = client
        self.model = model
        self.index_cls = index_cls
        
        # self.current_index = None


    def get_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        if "v=" in url:
            return url.split("v=")[1][:11]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1][:11]
        return url
    
    
    def get_summary(self, video_id: str) -> str:
        """Extract the summary of educational concepts from a transcript."""
        with open("data/transcripts.json", "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        for entry in transcripts:
            summary = entry.get(video_id).get("summary")
            print(summary)
            
        if not summary:
            return "Error: Summary not available for this video."
        return summary

    
    def search_video_transcript(self, search_query: str, video_id: str = None) -> str:
        """
        Performs a lexical search over the video's transcript to find specific explanations.

        Args:
            search_query (str): The search query.
            video_id (str): Uses video ID to filter results.

        Returns:
            str: JSON formatted search results.
        """

        if self.index_cls is None:
            return (
                "Error: No video has been processed yet. "
                "Please ask the user to provide a YouTube URL first."
            )

        filter_dict = {"video_id": video_id} if video_id else None

        print(
            f"Searching transcript for: '{search_query}' "
            f"with filter: {filter_dict}"
        )

        results = self.index_cls.search(
            search_query,
            filter_dict=filter_dict,
            num_results=5
        )

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
