import os
import json
import requests
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re

load_dotenv()

class EnhancedWebsiteFileGenerator:
    def __init__(self, 
                 output_dir: str = 'generated_website', 
                 brave_api_key: str = None,
                 template_dir: str = 'website_templates'):
        """
        Initialize the enhanced website file generator
        
        :param output_dir: Directory where website files will be saved
        :param brave_api_key: API key for Brave Search
        :param template_dir: Directory for website templates
        """
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.claude_reviewer = ChatAnthropic(model='claude-3-5-haiku-20241022', temperature=0.7)
        self.output_dir = output_dir
        self.brave_api_key = brave_api_key
        self.template_dir = template_dir
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.template_dir, exist_ok=True)
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform web search using Brave API
        
        :param query: Search query
        :param num_results: Number of search results to return
        :return: List of search results
        """
        if not self.brave_api_key:
            print("Brave API key not provided. Skipping web search.")
            return []
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": self.brave_api_key,
            "Accept": "application/json"
        }
        params = {
            "q": query,
            "count": num_results
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get('web', {}).get('results', [])
        except requests.RequestException as e:
            print(f"Web search error: {e}")
            return []
    
    def get_website_template(self, category: str = 'default') -> str:
        """
        Retrieve a website template with enhanced CodePen search capabilities
        
        :param category: Template category (e.g., 'portfolio', 'blog', 'business')
        :return: Template content
        """
        # First, try to extract keywords from the most recent user input
        try:
            keywords_prompt = PromptTemplate.from_template(
                "Extract 3-5 most relevant keywords from the following website description "
                "that would help find a matching CodePen template:\n\n"
                "Description: {description}\n\n"
                "Focus on design style, layout type, and key visual characteristics. "
                "Provide keywords that would be useful for searching CodePen templates."
            )
            
            # Assuming the last user input is stored or can be retrieved
            last_user_input = getattr(self, '_last_user_input', 'modern responsive website')
            
            keywords_chain = keywords_prompt | self.llm
            keywords_result = keywords_chain.invoke({"description": last_user_input})
            extracted_keywords = str(keywords_result.content).strip()
        except Exception:
            extracted_keywords = category
        
        # Perform Brave API search for CodePen templates
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "X-Subscription-Token": self.brave_api_key,
                "Accept": "application/json"
            }
            params = {
                "q": f"site:codepen.io {extracted_keywords} template html css",
                "count": 5
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json().get('web', {}).get('results', [])
            
            # Try to extract template from CodePen URLs
            for result in search_results:
                codepen_url = result.get('url', '')
                if 'codepen.io' in codepen_url:
                    try:
                        # Fetch CodePen page
                        codepen_page = requests.get(codepen_url)
                        codepen_page.raise_for_status()
                        
                        # Basic parsing to extract template
                        soup = BeautifulSoup(codepen_page.text, 'html.parser')
                        
                        # Look for template-like elements
                        template_content = soup.find('body')
                        if template_content:
                            # Construct a basic template
                            full_template = f"""
                            <!DOCTYPE html>
                            <html lang="en">
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <title>CodePen Inspired Template</title>
                                <style>
                                {str(soup.find('style')) if soup.find('style') else '/* Custom CSS */'}
                                </style>
                            </head>
                            <body>
                            {template_content}
                            <script>
                            {str(soup.find('script')) if soup.find('script') else '// Custom JavaScript'}
                            </script>
                            </body>
                            </html>
                            """
                            return full_template
                    except Exception:
                        continue
            
            # Fallback to existing template retrieval
            template_path = os.path.join(self.template_dir, f"{category}_template.html")
            
            # If specific template doesn't exist, use default
            if not os.path.exists(template_path):
                template_path = os.path.join(self.template_dir, "default_template.html")
            
            # If no template exists, return empty string
            if not os.path.exists(template_path):
                return ""
            
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        except Exception:
            # Final fallback
            template_path = os.path.join(self.template_dir, f"{category}_template.html")
            
            if not os.path.exists(template_path):
                template_path = os.path.join(self.template_dir, "default_template.html")
            
            if not os.path.exists(template_path):
                return ""
            
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def generate_html(self, content_description: str, search_context: List[Dict] = None) -> str:
        """
        Generate HTML content with optional web search context
        
        :param content_description: Description of website content
        :param search_context: Optional web search results for context
        :return: Generated HTML content
        """
        # Get a base template
        template = self.get_website_template()
        
        # Prepare context from web search
        context_str = ""
        if search_context:
            context_str = "\n".join([
                f"Web Search Context {i+1}: {result.get('title', '')} - {result.get('description', '')}" 
                for i, result in enumerate(search_context[:3])
            ])
        
        html_prompt = PromptTemplate.from_template(
            "Create a complete, modern HTML5 structure for a website with:\n"
            "Content Description: {content_description}\n"
            "Web Search Context:\n{context_str}\n\n"
            "Base Template:\n{template}\n\n"
            "Requirements:\n"
            "- Use semantic HTML5 tags\n"
            "- Include responsive design meta tags\n"
            "- Create image placeholders using https://picsum.photos/\n"
            "- Integrate web search context creatively\n"
            "- Maintain a clean, modern layout"
        )
        
        html_chain = html_prompt | self.llm
        html_result = html_chain.invoke({
            "content_description": content_description,
            "context_str": context_str,
            "template": template
        })
        
        return str(html_result.content)
    
    def generate_css(self, design_description: str) -> str:
        """
        Generate CSS content based on the design description
        
        :param design_description: Description of website design
        :return: Generated CSS content
        """
        css_prompt = PromptTemplate.from_template(
            "Create a comprehensive CSS file with a modern, responsive design based on:\n"
            "{design_description}\n\n"
            "Requirements:\n"
            "- Use CSS Grid or Flexbox for layout\n"
            "- Include media queries for responsiveness\n"
            "- Create a clean, professional color scheme\n"
            "- Implement subtle animations\n"
            "- Ensure cross-browser compatibility"
        )
        
        css_chain = css_prompt | self.llm
        css_result = css_chain.invoke({"design_description": design_description})
        
        return str(css_result.content)
    
    def generate_javascript(self, features_description: str) -> str:
        """
        Generate JavaScript content based on feature description
        
        :param features_description: Description of website features
        :return: Generated JavaScript content
        """
        js_prompt = PromptTemplate.from_template(
            "Create a JavaScript file to enhance the website with interactive features based on:\n"
            "{features_description}\n\n"
            "Requirements:\n"
            "- Use modern ES6+ syntax\n"
            "- Add dynamic interactivity\n"
            "- Implement error handling\n"
            "- Ensure cross-browser compatibility\n"
            "- Include performance optimization techniques"
        )
        
        js_chain = js_prompt | self.llm
        js_result = js_chain.invoke({"features_description": features_description})
        
        return str(js_result.content)
    
    def save_website_files(self, 
                            html_content: str, 
                            css_content: str, 
                            js_content: str) -> Dict[str, str]:
        """
        Save generated website files to the output directory
        
        :param html_content: HTML file content
        :param css_content: CSS file content
        :param js_content: JavaScript file content
        :return: Dictionary of file paths
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File paths
        html_path = os.path.join(self.output_dir, f'index_{timestamp}.html')
        css_path = os.path.join(self.output_dir, f'styles_{timestamp}.css')
        js_path = os.path.join(self.output_dir, f'script_{timestamp}.js')
        
        # Write files
        with open(html_path, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)
        
        with open(css_path, 'w', encoding='utf-8') as css_file:
            css_file.write(css_content)
        
        with open(js_path, 'w', encoding='utf-8') as js_file:
            js_file.write(js_content)
        
        # Modify HTML to link CSS and JS files
        modified_html = self._link_css_js(html_content, 
                                          os.path.basename(css_path), 
                                          os.path.basename(js_path))
        
        # Overwrite HTML with linked version
        with open(html_path, 'w', encoding='utf-8') as html_file:
            html_file.write(modified_html)
        
        return {
            'html': html_path,
            'css': css_path,
            'js': js_path
        }
    
    def _link_css_js(self, html_content: str, css_filename: str, js_filename: str) -> str:
        """
        Modify HTML to include CSS and JS file links
        
        :param html_content: Original HTML content
        :param css_filename: CSS filename
        :param js_filename: JavaScript filename
        :return: Modified HTML content
        """
        css_link = f'<link rel="stylesheet" href="{css_filename}">'
        html_content = html_content.replace('</head>', f'{css_link}</head>')
        
        js_script = f'<script src="{js_filename}"></script>'
        html_content = html_content.replace('</body>', f'{js_script}</body>')
        
        return html_content
    
    def review_and_improve_website(self, 
                                   html_content: str, 
                                   css_content: str, 
                                   js_content: str, 
                                   user_input: str) -> Dict[str, str]:
        """
        Use Claude to review and improve the generated website files
        
        :param html_content: Generated HTML content
        :param css_content: Generated CSS content
        :param js_content: Generated JavaScript content
        :param user_input: Original user requirements
        :return: Improved website content
        """
        review_prompt = PromptTemplate.from_template(
            "Comprehensively review the following website components generated based on user requirements: {user_input}\n\n"
            "HTML Content:\n{html_content}\n\n"
            "CSS Content:\n{css_content}\n\n"
            "JavaScript Content:\n{js_content}\n\n"
            "Provide an in-depth analysis and SPECIFIC, ACTIONABLE improvements focusing on:\n"
            "1. Semantic HTML structure and accessibility\n"
            "2. CSS optimization and responsiveness\n"
            "3. JavaScript performance and functionality\n"
            "4. Best practices and modern web standards\n\n"
            "IMPORTANT: For EACH improvement, provide:\n"
            "- EXACT code changes\n"
            "- Rationale for the change\n"
            "- Specific line numbers or code blocks to modify\n"
            "Format your response with clear, implementable suggestions."
        )
        
        review_chain = review_prompt | self.claude_reviewer
        review_result = review_chain.invoke({
            "user_input": user_input,
            "html_content": html_content,
            "css_content": css_content,
            "js_content": js_content
        })
        
        # Parse Claude's response
        review_text = str(review_result.content)
        
        # Automatically apply improvements
        improved_html = self._apply_code_improvements(html_content, review_text, 'html')
        improved_css = self._apply_code_improvements(css_content, review_text, 'css')
        improved_js = self._apply_code_improvements(js_content, review_text, 'javascript')
        
        return {
            'html': improved_html or html_content,
            'css': improved_css or css_content,
            'js': improved_js or js_content,
            'review': review_text
        }
    
    def _apply_code_improvements(self, original_code: str, review_text: str, language: str) -> str:
        """
        Automatically apply improvements to the code based on Claude's review
        
        :param original_code: Original code content
        :param review_text: Claude's review text
        :param language: Language of the code (html, css, javascript)
        :return: Improved code
        """
        # Extract specific improvement suggestions
        improvement_pattern = r'(Improvement \d+:|Change:)\s*(.*?)\n\n'
        improvements = re.findall(improvement_pattern, review_text, re.DOTALL)
        
        improved_code = original_code
        
        for _, improvement in improvements:
            # Check if improvement is relevant to this language
            if not self._is_improvement_relevant(improvement, language):
                continue
            
            try:
                # Try to apply the improvement
                improved_code = self._apply_single_improvement(improved_code, improvement)
            except Exception as e:
                print(f"Could not apply improvement: {improvement[:100]}... Error: {e}")
        
        return improved_code
    
    def _is_improvement_relevant(self, improvement: str, language: str) -> bool:
        """
        Check if an improvement is relevant to the specific language
        
        :param improvement: Improvement suggestion text
        :param language: Target language (html, css, javascript)
        :return: Boolean indicating relevance
        """
        # Convert improvement and language to lowercase for case-insensitive matching
        improvement_lower = improvement.lower()
        language_lower = language.lower()
        
        # Keywords to identify language-specific improvements
        language_keywords = {
            'html': ['html', 'semantic', 'accessibility', 'tag', 'structure', 'aria'],
            'css': ['css', 'style', 'responsive', 'layout', 'media query', 'flexbox', 'grid'],
            'javascript': ['js', 'javascript', 'function', 'performance', 'event', 'optimize']
        }
        
        # Check if any keywords for the specific language are in the improvement
        return any(
            keyword in improvement_lower 
            for keyword in language_keywords.get(language_lower, [])
        )
    
    def _apply_single_improvement(self, code: str, improvement: str) -> str:
        """
        Apply a single specific improvement to the code
        
        :param code: Original code
        :param improvement: Specific improvement suggestion
        :return: Improved code
        """
        # Detect if the improvement suggests a replacement
        replacement_match = re.search(r'Replace\s*(?:line|code)?\s*(?:\d+)?:?\s*```(?:html|css|js)?\n(.*?)```', 
                                      improvement, 
                                      re.DOTALL | re.IGNORECASE)
        
        # If direct replacement found
        if replacement_match:
            replacement_code = replacement_match.group(1).strip()
            return replacement_code if replacement_code else code
        
        # Try to find and replace specific code segments
        for line in code.splitlines():
            if line.strip() in improvement:
                # Find a potential replacement
                replacement_line_match = re.search(r'Replace with:\s*(.+)', improvement)
                if replacement_line_match:
                    replacement_line = replacement_line_match.group(1).strip()
                    code = code.replace(line, replacement_line)
                    break
        
        return code
    
    def create_website(self, 
                       user_input: str, 
                       max_iterations: int = 3) -> Dict[str, Any]:
        """
        Full workflow to generate and save website files with iterative improvement
        
        :param user_input: User's website requirements
        :param max_iterations: Maximum number of improvement iterations
        :return: Dictionary of generated content and file paths
        """
        # Perform initial web search for context
        search_results = self.search_web(user_input)
        
        for iteration in range(max_iterations):
            # Generate components using GPT
            html_content = self.generate_html(user_input, search_results)
            css_content = self.generate_css(user_input)
            js_content = self.generate_javascript(user_input)
            
            # Review and potentially improve with Claude
            claude_review = self.review_and_improve_website(
                html_content, css_content, js_content, user_input
            )
            
            # Use Claude's improved versions
            html_content = claude_review['html']
            css_content = claude_review['css']
            js_content = claude_review['js']
            
            # Save files
            file_paths = self.save_website_files(html_content, css_content, js_content)
            
            # Present to user for feedback
            print(f"\n--- Iteration {iteration + 1} ---")
            print("Website generated. Preview the files in the generated locations.")
            
            # Print Claude's review
            print("\n--- Claude's Review ---")
            print(claude_review['review'])
            
            # Ask for user satisfaction
            user_satisfied = input("Are you satisfied with this website version? (yes/no): ").lower()
            
            if user_satisfied in ['yes', 'y']:
                return {
                    'input': user_input,
                    'html_content': html_content,
                    'css_content': css_content,
                    'js_content': js_content,
                    'file_paths': file_paths,
                    'claude_review': claude_review['review']
                }
            
            # Get user feedback for improvement
            user_input = input("Please provide additional details or changes you'd like: ")
        
        print("Maximum iterations reached. Final version saved.")
        return {
            'input': user_input,
            'html_content': html_content,
            'css_content': css_content,
            'js_content': js_content,
            'file_paths': file_paths,
            'claude_review': claude_review['review']
        }


def main():
    # Create website generator (replace with your actual Brave API key)
    generator = EnhancedWebsiteFileGenerator(
        output_dir='my_websites', 
        brave_api_key=os.getenv('BRAVE_API_KEY')
    )

    # Get user input
    user_input = input("Enter your website requirements: ")
    
    # Generate website
    result = generator.create_website(user_input)
    
    # Print out file locations
    print("Website Generated Successfully!")
    print("\nGenerated Files:")
    for key, value in result['file_paths'].items():
        print(f"{key.upper()}: {value}")
    
    # Optional: Print contents
    print("\n--- HTML Content Preview ---")
    print(result['html_content'][:500] + "...")
    
    print("\n--- CSS Content Preview ---")
    print(result['css_content'][:500] + "...")
    
    print("\n--- JS Content Preview ---")
    print(result['js_content'][:500] + "...")

if __name__ == "__main__":
    main()



