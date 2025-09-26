import os
import json
import logging
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Your Proven Prompt ---
# Encapsulated in a function for clarity
def get_generation_prompt():
    """Returns the detailed prompt for generating the JSON knowledge base."""
    return """
Your mission is to analyze the entire attached PDF document, page by page, and generate a single, valid JSON array. This array must contain one JSON object for each page of the document.

The `content` field is your primary output. It is not a summary. It must be a **complete textual representation of everything on the slide.**

**Instructions for each JSON object:**

For each page in the PDF, create a JSON object with the following fields:

1.  **`page_number`** (integer): The accurate page number.
2.  **`slide_title`** (string): The main title or heading of the slide. If there is no title, use "Untitled".
3.  **`content_type`** (string): The primary format of the page. Choose from one of the following: `diagram`, `table`, `list`, `title_page`, `text_only`.
4.  **`content`** (string): **This is the most critical field. It must fully contain all information on the slide without summarization.** To achieve this, you will:
    *   First, state the slide's title and any introductory text.
    *   Next, describe the visual structure of the slide (e.g., "The information is organized into six distinct boxes in a grid," "This is a circular diagram with a central hub and six connected points").
    *   Then, systematically go through each visual element (box, list item, diagram component).
    *   **CRITICAL INSTRUCTION: For each element, you MUST explicitly state its title or label, followed immediately by its full, verbatim text content. You must create explicit "key-value" pairs in your description. Do NOT list all titles and then all descriptions separately.**
    *   **Crucially, do not paraphrase, interpret, or shorten any text found on the slide.** You are creating the definitive textual version of the visual slide.
5.  **`image_path`** (string): The local file path for the image of this page, following the format `./images/page_N.png` where N is the page number.

---

**One-Shot Example for Page 3 (Follow this format precisely):**

```json
{
  "page_number": 3,
  "slide_title": "Current Status",
  "content_type": "diagram",
  "content": "The slide is titled 'Current Status'. It has an introductory sentence: 'Our assessment revealed key communication gaps, underscoring the need for aligned narratives, stronger media presence, and a unified internal voice to position Ashraq as a strategic enabler..'. The layout presents six key gaps in a diagrammatic format. Each gap is presented in a separate box with a title and its full text description. The six gaps are: 1. Box titled 'Limited Public Awareness', with text: 'Ashraq is often mistaken for a real estate developer. Its enabler role needs clearer positioning to reshape stakeholder perception'. 2. Box titled 'Inconsistent Messaging', with text: 'Messaging varies across platforms and audiences, lacking a unified and consistent narrative.'. 3. Box titled 'Low Media & Executive Visibility', with text: 'Limited storytelling and minimal leadership exposure reduce Ashraq's visibility and perceived authority.'. 4. Box titled 'Narrow Content Scope', with text: 'Focus is largely on investment—missing opportunities to highlight transformation, social impact, and people-centric success stories.'. 5. Box titled 'Internal Gaps & Communication Silos', with text: 'Communication is informal across teams, creating confusion around priorities, updates, and alignment.'. 6. Box titled 'Missed Digital Opportunities', with text: 'Digital presence lacks human-centered content and interactive storytelling that build trust and community engagement'.",
  "image_path": "./images/page_3.png"
}
```

---

**Final Output Requirement:**

Generate the complete JSON array for all pages. **Your output must be ONLY the JSON array itself, starting with `[` and ending with `]`**. Do not include any introductory text, explanations, or markdown formatting like `json` tags.
"""

class KnowledgeBaseGenerator:
    """
    Generates a structured JSON knowledge base from a PDF document using a multimodal LLM.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        """
        Initializes the generator with the Google Gemini API key.
        """
        if not api_key:
            raise ValueError("Google Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logging.info(f"Initialized Generative Model: {model_name}")

    def _pdf_to_images(self, pdf_path: Path) -> list[Image.Image]:
        """Converts each page of a PDF into a PIL Image object."""
        images = []
        try:
            pdf_document = fitz.open(pdf_path)
            logging.info(f"Processing {len(pdf_document)} pages from PDF: {pdf_path.name}")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                # Render page to a pixmap (image) at a higher resolution
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
            pdf_document.close()
            return images
        except Exception as e:
            logging.error(f"Failed to convert PDF to images: {e}")
            raise

    def generate(self, pdf_path: Path, output_path: Path) -> bool:
        """
        Generates the data.json file from the given PDF.

        Args:
            pdf_path: The path to the source PDF file.
            output_path: The path to save the generated data.json file.

        Returns:
            True if generation was successful, False otherwise.
        """
        if not pdf_path.exists():
            logging.error(f"PDF file not found at: {pdf_path}")
            return False

        try:
            logging.info("Step 1: Converting PDF to images...")
            page_images = self._pdf_to_images(pdf_path)
            
            logging.info("Step 2: Preparing content for the vision model...")
            prompt = get_generation_prompt()
            # The content list should contain the prompt first, then all images
            content_for_llm = [prompt] + page_images

            logging.info("Step 3: Sending request to Gemini...")
            response = self.model.generate_content(content_for_llm)
            
            logging.info("Step 4: Processing and validating the response...")
            # Clean up the response text to ensure it's valid JSON
            json_text = response.text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            # Parse the JSON
            parsed_json = json.loads(json_text)
            
            # Basic validation
            if not isinstance(parsed_json, list) or len(parsed_json) != len(page_images):
                raise ValueError(f"Validation failed: Expected a list of {len(page_images)} items.")
            logging.info(f"Successfully parsed and validated JSON for {len(parsed_json)} pages.")
            
            logging.info(f"Step 5: Saving knowledge base to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            
            logging.info("✅ Knowledge base generation complete!")
            return True

        except Exception as e:
            logging.error(f"An error occurred during knowledge base generation: {e}")
            return False

if __name__ == '__main__':
    # This allows the script to be run directly for testing or manual generation
    from dotenv import load_dotenv
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("FATAL: GOOGLE_API_KEY environment variable not set.")
        print("Please add it to your .env file.")
    else:
        # Define paths relative to the project root
        project_root = Path(__file__).parent.parent
        pdf_file = project_root / "data" / "task-mohamed-rag.pdf"
        json_output_file = project_root / "data.json"

        generator = KnowledgeBaseGenerator(api_key=google_api_key)
        generator.generate(pdf_path=pdf_file, output_path=json_output_file)
