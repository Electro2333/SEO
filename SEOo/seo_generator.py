import json
import requests
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import re
from datetime import datetime
import aiohttp
import os
from openai import OpenAI
import asyncio
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
@dataclass
class Config:
    model: str = "mistralai/mistral-7b-instruct:free"
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    site_url: str = "https://yoursiteurl.com"  # Update with your site URL
    site_name: str = "Your Site Name"  # Update with your site name
    model_fallbacks: list = field(default_factory=lambda: [
        "meta-llama/llama-3.3-70b-instruct:free",
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "deepseek/deepseek-r1:free",
        "google/gemini-2.0-flash-thinking-exp:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "qwen/qwen-vl-plus:free",
        "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "sophosympatheia/rogue-rose-103b-v0.2:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "gryphe/mythomax-l2-13b:free"
    ])

@dataclass
class SEOParams:
    min_word_count: int = 300
    max_word_count: int = 5000
    content_type: str = "article"
    target_audience: str = "general"
    tone: str = "professional"
    language: str = "en"
    include_images: bool = True
    content_structure: str = "default"
    max_paragraph_length: int = 150
    heading_frequency: int = 250
    keyword_density: float = 0.02
    lsi_keyword_count: int = 10
    include_meta_tags: bool = True
    include_faq_schema: bool = True
    include_howto_schema: bool = True
    include_product_schema: bool = False
    add_open_graph: bool = True
    add_twitter_card: bool = True
    readability_target: str = "grade8"
    media_placeholders: bool = True
    internal_link_count: int = 5
    external_link_count: int = 3
    table_toc: bool = True
    call_to_action: bool = True
    entity_optimization: bool = True
    semantic_html5: bool = True
    product_comparison: bool = True
    spec_table_format: bool = True
    pros_cons_format: bool = True
    rating_system: bool = True
    price_tracking: bool = True
    benchmark_scores: bool = True
    user_experience_section: bool = True

@dataclass
class ContentType:
    key: str
    name: str
    icon: str
    tones: List[str]
    audiences: List[str]
    languages: List[str]
    structure: Dict[str, List[str]]
    prompt_template: str
    seo_weight: float  # SEO importance weight for this content type

CONTENT_TYPES = {
    "article": ContentType(
        key="article",
        name="Article",
        icon="📄",
        tones=["professional", "conversational", "academic", "journalistic", "technical"],
        audiences=["general", "experts", "professionals", "students", "researchers"],
        languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        structure={
            "sections": [
                "Executive Summary",
                "Introduction",
                "Background",
                "Main Analysis",
                "Expert Insights",
                "Research Findings",
                "Future Implications",
                "Conclusion"
            ],
            "elements": ["statistics", "expert_quotes", "case_studies", "research_data"]
        },
        prompt_template="""
        Create a comprehensive {tone} article about {topic} for {audience} audience.
        Language: {language}
        Target length: {word_count} words
        
        Structure Requirements:
        1. Start with an intriguing hook/question/statistic
        2. Provide historical context and current relevance
        3. Include 3-5 actionable insights with real-world examples
        4. Add expert quotes and recent research findings
        5. Incorporate data visualizations (charts/graphs) concepts
        6. Address common misconceptions
        7. Conclude with practical applications
        
        Engagement Elements:
        - Use storytelling techniques
        - Include rhetorical questions
        - Add "Pro Tip" boxes with insider knowledge
        - Use analogies for complex concepts
        - Incorporate case studies
        
        SEO Requirements:
        - Natural keyword integration (1-2% density)
        - Semantic variations of main keywords
        - FAQ section with 5-7 questions
        - Internal linking suggestions
        - Mobile-friendly formatting
        """,
        seo_weight=0.95
    ),
    "product_review": ContentType(
        key="product_review",
        name="Product Review",
        icon="⭐",
        tones=["professional", "conversational", "enthusiastic"],
        audiences=["general", "experts", "enthusiasts"],
        languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        structure={
            "sections": [
                "Introduction",
                "Product Overview",
                "Features and Specifications",
                "Performance Analysis",
                "Pros and Cons",
                "Benchmark Comparison",
                "Conclusion",
                "FAQ"
            ],
            "elements": ["product_images", "spec_table", "comparison_table", "user_reviews"]
        },
        prompt_template="""
        Create a detailed {tone} product review for {topic} aimed at {audience} audience.
        Language: {language}
        Target length: {word_count} words

        Required elements:
        - A clear product overview that highlights how the product compares with industry benchmarks.
        - Detailed features and specifications with a dedicated benchmark comparison section.
        - Performance analysis with compelling language that emphasizes an excellent user experience.
        - A balanced pros and cons section.
        - User reviews and ratings to provide social proof.
        - Clear calls-to-action urging readers to learn more or try the product.
        - An FAQ section addressing common queries.

        Style guidelines:
        - Maintain a consistent, {tone} tone throughout.
        - Use engaging and persuasive language tailored for {audience} readers.
        - Incorporate natural keyword integration without disrupting the flow.
        """,
        seo_weight=0.8
    ),
    "how_to": ContentType(
        key="how_to",
        name="How-To Guide",
        icon="📖",
        tones=["instructional", "friendly", "step-by-step"],
        audiences=["general", "beginners", "enthusiasts"],
        languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        structure={
            "sections": [
                "Introduction",
                "Materials Needed",
                "Step-by-Step Instructions",
                "Tips and Tricks",
                "Troubleshooting",
                "Conclusion"
            ],
            "elements": ["step_images", "video_tutorial", "checklist"]
        },
        prompt_template="""
        Create a comprehensive {tone} how-to guide on {topic} for {audience} audience.
        Language: {language}
        Target length: {word_count} words
        
        Required elements:
        - Clear introduction
        - List of materials needed
        - Detailed step-by-step instructions
        - Tips and tricks
        - Troubleshooting section
        - Conclusion
        
        Style guidelines:
        - Maintain {tone} tone throughout
        - Use simple language suitable for {audience}
        - Include relevant images or video tutorials
        - Optimize for {language} SEO
        """,
        seo_weight=0.7
    ),
    "comparison": ContentType(
        key="comparison",
        name="Comparison",
        icon="⚖️",
        tones=["analytical", "objective", "informative"],
        audiences=["general", "experts", "decision-makers"],
        languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        structure={
            "sections": [
                "Introduction",
                "Product/Service Overview",
                "Comparison Criteria",
                "Detailed Comparison",
                "Pros and Cons",
                "Conclusion",
                "Recommendation"
            ],
            "elements": ["comparison_table", "feature_list", "rating_system"]
        },
        prompt_template="""
        Create a thorough {tone} comparison of {topic} for {audience} audience.
        Language: {language}
        Target length: {word_count} words
        
        Required elements:
        - Introduction to the products/services being compared
        - Clear comparison criteria
        - Detailed comparison of features and performance
        - Pros and cons of each option
        - Conclusion and recommendation
        
        Style guidelines:
        - Maintain {tone} tone throughout
        - Use appropriate terminology for {audience}
        - Include a comparison table
        - Optimize for {language} SEO
        """,
        seo_weight=0.8
    ),
    "listicle": ContentType(
        key="listicle",
        name="Listicle",
        icon="📋",
        tones=["engaging", "informative", "entertaining"],
        audiences=["general", "enthusiasts", "curious"],
        languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        structure={
            "sections": [
                "Introduction",
                "List Items",
                "Conclusion"
            ],
            "elements": ["numbered_list", "bullet_points", "images"]
        },
        prompt_template="""
        Create an engaging {tone} listicle about {topic} for {audience} audience.
        Language: {language}
        Target length: {word_count} words
        
        Required elements:
        - Brief introduction
        - Numbered or bullet-pointed list of items
        - Conclusion summarizing key points
        
        Style guidelines:
        - Maintain {tone} tone throughout
        - Use catchy titles for each list item
        - Include relevant images
        - Optimize for {language} SEO
        """,
        seo_weight=0.6
    ),
    "case_study": ContentType(
        key="case_study",
        name="Case Study",
        icon="🔬",
        tones=["professional", "analytical", "informative"],
        audiences=["professionals", "researchers", "students"],
        languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        structure={
            "sections": [
                "Introduction",
                "Background",
                "Methodology",
                "Results",
                "Analysis",
                "Conclusion",
                "Recommendations"
            ],
            "elements": ["data_tables", "graphs", "quotes"]
        },
        prompt_template="""
        Create a detailed {tone} case study on {topic} for {audience} audience.
        Language: {language}
        Target length: {word_count} words
        
        Required elements:
        - Introduction to the case
        - Background information
        - Methodology used
        - Results and findings
        - Analysis of the data
        - Conclusion and key takeaways
        - Recommendations for future action
        
        Style guidelines:
        - Maintain {tone} tone throughout
        - Use appropriate terminology for {audience}
        - Include relevant data visualizations
        - Optimize for {language} SEO
        """,
        seo_weight=0.9
    ),
    "news_article": ContentType(
        key="news_article",
        name="News Article",
        icon="📰",
        tones=["journalistic", "objective", "informative"],
        audiences=["general", "news_enthusiasts", "professionals"],
        languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        structure={
            "sections": [
                "Headline",
                "Lead Paragraph",
                "Body",
                "Quotes",
                "Conclusion"
            ],
            "elements": ["images", "infographics", "sidebars"]
        },
        prompt_template="""
        Create a {tone} news article about {topic} for {audience} audience.
        Language: {language}
        Target length: {word_count} words
        
        Required elements:
        - Attention-grabbing headline
        - Concise lead paragraph summarizing the story
        - Detailed body with facts and context
        - Relevant quotes from sources
        - Conclusion summarizing key points
        
        Style guidelines:
        - Maintain {tone} tone throughout
        - Use clear, concise language suitable for {audience}
        - Include relevant images or infographics
        - Optimize for {language} SEO
        """,
        seo_weight=0.7
    ),
    "easy_learn": ContentType(
        key="easy_learn",
        name="Easy Learn",
        icon="🎓",
        tones=["friendly", "analogical", "simplified"],
        audiences=["students", "learners", "professors", "curious"],
        languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        structure={
            "sections": [
                "Introduction",
                "Main Concept",
                "Analogy",
                "Examples",
                "Key Takeaways",
                "Further Reading"
            ],
            "elements": ["illustrations", "diagrams", "real-life_examples"]
        },
        prompt_template="""
        Create a {tone} educational piece on {topic} for {audience} audience.
        Language: {language}
        Target length: {word_count} words
        
        Required elements:
        - Brief introduction to the topic
        - Clear explanation of the main concept
        - An analogy to help understand the concept
        - Real-life examples to illustrate the concept
        - Key takeaways summarizing the main points
        - Suggestions for further reading or exploration
        
        Style guidelines:
        - Maintain {tone} tone throughout
        - Use simple language suitable for {audience}
        - Include relevant illustrations or diagrams
        - Optimize for {language} SEO
        """,
        seo_weight=0.6
    )
}

@dataclass
class ContentParameters:
    content_type: str
    target_audience: str
    tone: str
    language: str
    word_count: int
    seo_optimization_level: float = 1.0
    reasoning_iterations: int = 3  # Number of reasoning passes
    verification_depth: int = 2   # How thoroughly to verify content
    simplification_level: int = 1  # How much to simplify complex concepts
    
    def validate(self) -> bool:
        """Validate content parameters against allowed values."""
        content_type_obj = CONTENT_TYPES.get(self.content_type)
        if not content_type_obj:
            return False
            
        return (
            self.target_audience in content_type_obj.audiences and
            self.tone in content_type_obj.tones and
            self.language in content_type_obj.languages and
            300 <= self.word_count <= 5000
        )

class ContentGenerationError(Exception):
    """Custom exception for content generation failures"""
    pass

class ContentOptimizer:
    def __init__(self, parameters: ContentParameters):
        self.parameters = parameters
        self.content_type = CONTENT_TYPES[parameters.content_type]
        
    def optimize_for_audience(self, content: str) -> str:
        """Optimize content for specific audience."""
        audience_patterns = {
            "general": {
                "readability_level": "grade8",
                "technical_terms": "minimal",
                "explanation_depth": "moderate"
            },
            "experts": {
                "readability_level": "technical",
                "technical_terms": "extensive",
                "explanation_depth": "deep"
            },
            # Add other audience patterns
        }
        
        pattern = audience_patterns.get(self.parameters.target_audience, audience_patterns["general"])
        # Apply audience-specific optimizations
        return self._apply_audience_patterns(content, pattern)
        
    def optimize_for_tone(self, content: str) -> str:
        """Optimize content tone."""
        tone_patterns = {
            "professional": {
                "formality": "high",
                "terminology": "industry-standard",
                "structure": "formal"
            },
            "conversational": {
                "formality": "low",
                "terminology": "simple",
                "structure": "flexible"
            },
            # Add other tone patterns
        }
        
        pattern = tone_patterns.get(self.parameters.tone, tone_patterns["professional"])
        # Apply tone-specific optimizations
        return self._apply_tone_patterns(content, pattern)
        
    def optimize_for_language(self, content: str) -> str:
        """Optimize content for specific language."""
        # Apply language-specific SEO optimizations
        return self._apply_language_optimizations(content, self.parameters.language)

    def _apply_audience_patterns(self, content: str, pattern: Dict) -> str:
        """Apply audience-specific optimization patterns."""
        # Implementation of audience-specific optimizations
        return content

    def _apply_tone_patterns(self, content: str, pattern: Dict) -> str:
        """Apply tone-specific optimization patterns."""
        # Implementation of tone-specific optimizations
        return content

    def _apply_language_optimizations(self, content: str, language: str) -> str:
        """Apply language-specific optimizations."""
        # Implementation of language-specific optimizations
        return content

class SEOContentGenerator:
    def __init__(self, config: Config = Config(), seo_params: SEOParams = SEOParams()):
        self.config = config
        self.seo_params = seo_params
        self.weights = {
            "keyword_optimization": 0.3,
            "content_quality": 0.25,
            "readability": 0.2,
            "structure": 0.1,
            "technical_seo": 0.1,
            "user_experience": 0.05
        }
        
    async def generate_seo_content(self, topic: str, keywords: List[str], parameters: ContentParameters, options: dict = None) -> Dict:
        """Generate SEO-optimized content with proper async handling"""
        try:
            # Generate initial draft
            draft = await self._generate_initial_draft(topic, keywords, parameters)
            
            # Multi-step reasoning process
            for iteration in range(parameters.reasoning_iterations):
                analysis = self._analyze_content(draft, keywords, parameters)
                draft = await self._refine_content(draft, analysis, parameters)
                
                if parameters.verification_depth > 0:
                    verification = self._verify_content(draft, parameters)
                    draft = await self._apply_verification(draft, verification, parameters)

            # Final optimization
            optimized_content = await self._optimize_for_engagement(draft, parameters)
            
            return {
                "content": optimized_content,
                "seo_score": self._calculate_seo_score(optimized_content, keywords, parameters),
                "metadata": self._generate_metadata(topic, optimized_content, parameters)
            }
            
        except Exception as e:
            raise ContentGenerationError(f"Generation failed: {str(e)}") from e

    async def _generate_initial_draft(self, topic: str, keywords: List[str], parameters: ContentParameters) -> str:
        """Generate initial content draft"""
        prompt = self._create_base_prompt(topic, keywords, parameters)
        return await self._generate_with_fallback(prompt, parameters)

    async def _refine_content(self, content: str, analysis: Dict, parameters: ContentParameters) -> str:
        """Refine content based on analysis"""
        prompt = self._create_refinement_prompt(content, analysis, parameters)
        return await self._generate_with_fallback(prompt, parameters)

    async def _apply_verification(self, content: str, verification: Dict, parameters: ContentParameters) -> str:
        """Apply verification corrections"""
        prompt = self._create_verification_prompt(content, verification)
        return await self._generate_with_fallback(prompt, parameters)

    async def _optimize_for_engagement(self, content: str, parameters: ContentParameters) -> str:
        """Final engagement optimization pass"""
        prompt = self._create_engagement_prompt(content, parameters)
        return await self._generate_with_fallback(prompt, parameters)

    def _analyze_content(self, content: str, keywords: List[str], parameters: ContentParameters) -> Dict:
        """Analyze content for improvement opportunities"""
        return {
            "complexity_issues": self._find_complex_sections(content),
            "seo_opportunities": self._identify_seo_opportunities(content, keywords),
            "engagement_gaps": self._detect_engagement_gaps(content),
            "structural_issues": self._analyze_content_structure(content)
        }

    def _verify_content(self, content: str, parameters: ContentParameters) -> Dict:
        """Verify content accuracy and consistency"""
        return {
            "fact_consistency": self._check_fact_consistency(content),
            "argument_flow": self._analyze_argument_flow(content),
            "conceptual_accuracy": self._verify_conceptual_accuracy(content),
            "style_consistency": self._check_style_consistency(content)
        }

    async def _generate_with_fallback(self, prompt: str, parameters: ContentParameters) -> str:
        """Generate content with model fallback support"""
        models = [self.config.model] + self.config.model_fallbacks
        client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        
        for attempt, model in enumerate(models, 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    timeout=30
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == len(models):
                    raise ContentGenerationError(
                        f"All models failed: {str(e)}"
                    ) from e
                continue
        return ""

    def _calculate_seo_score(self, content: str, keywords: List[str], parameters: ContentParameters) -> Dict:
        """Calculate SEO score with verification"""
        initial_score = {
            "overall": 0.0,
            "factors": {
                "keyword_optimization": self._analyze_keyword_usage(content, keywords),
                "content_quality": self._analyze_content_quality(content),
                "readability": self._analyze_readability(content),
                "structure": self._analyze_content_structure(content),
                "technical_seo": self._analyze_technical_seo(content),
                "user_experience": self._analyze_user_experience(content)
            }
        }
        
        # Cross-verify scores
        verification = {
            "keyword_validation": self._verify_keyword_usage(content, keywords),
            "content_depth_check": self._verify_content_depth(content),
            "semantic_coherence": self._analyze_semantic_coherence(content)
        }
        
        # Adjust scores based on verification
        adjustment_factors = {
            "keyword_optimization": 0.9 if verification["keyword_validation"] else 1.1,
            "content_depth": 1.2 if verification["content_depth_check"] else 0.8,
            "semantic_richness": 1.1 if verification["semantic_coherence"] else 0.9
        }
        
        # Apply adjustments
        for factor in initial_score["factors"]:
            initial_score["factors"][factor] *= adjustment_factors.get(factor, 1.0)
        
        # Recalculate overall score
        initial_score["overall"] = sum(
            initial_score["factors"][factor] * self.weights[factor] 
            for factor in initial_score["factors"]
        )
        
        return initial_score

    def _analyze_keyword_usage(self, content: str, keywords: List[str]) -> float:
        """Analyze keyword usage and optimization."""
        if not content or not keywords:
            return 0.0
        
        content_lower = content.lower()
        word_count = len(content.split())
        
        scores = []
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Calculate keyword density
            keyword_count = content_lower.count(keyword_lower)
            density = keyword_count / word_count if word_count > 0 else 0
            
            # Ideal density is between 1-3%
            density_score = min(1.0, density * 33.33 if 0.01 <= density <= 0.03 else 0.5)
            
            # Check keyword placement
            placement_score = 0.0
            if keyword_lower in content_lower[:200].lower():  # Beginning of content
                placement_score += 0.4
            if keyword_lower in content_lower[-200:].lower():  # End of content
                placement_score += 0.2
            
            # Check for keyword variations
            variations_score = self._check_keyword_variations(content_lower, keyword_lower)
            
            # Combine scores
            keyword_score = (density_score * 0.4 + placement_score * 0.3 + variations_score * 0.3)
            scores.append(keyword_score)
        
        return sum(scores) / len(scores) if scores else 0.0

    def _analyze_content_quality(self, content: str) -> float:
        """Analyze content quality metrics."""
        if not content:
            return 0.0
        
        scores = []
        
        # Length score
        word_count = len(content.split())
        length_score = min(1.0, word_count / 2000)  # Optimal length around 2000 words
        scores.append(length_score * 0.3)
        
        # Paragraph structure
        paragraphs = content.split('\n\n')
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        paragraph_score = 1.0 if 50 <= avg_paragraph_length <= 150 else 0.5
        scores.append(paragraph_score * 0.2)
        
        # Header usage
        header_pattern = r'<h[1-6]>.*?</h[1-6]>'
        headers_count = len(re.findall(header_pattern, content, re.IGNORECASE))
        header_score = min(1.0, headers_count / 6)  # Optimal: at least 6 headers
        scores.append(header_score * 0.2)
        
        # Media elements
        media_score = 0.0
        if '<img' in content.lower():
            media_score += 0.5
        if '<video' in content.lower() or '<iframe' in content.lower():
            media_score += 0.5
        scores.append(media_score * 0.15)
        
        # Internal/external links
        link_pattern = r'<a\s+(?:[^>]*?\s+)?href=(["\'])(.*?)\1'
        links_count = len(re.findall(link_pattern, content, re.IGNORECASE))
        link_score = min(1.0, links_count / 5)  # Optimal: at least 5 links
        scores.append(link_score * 0.15)
        
        return sum(scores)

    def _analyze_readability(self, content: str) -> float:
        """Analyze content readability."""
        if not content:
            return 0.0
        
        # Simplified Flesch-Kincaid calculation
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Flesch-Kincaid Grade Level
        grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        
        # Convert grade level to score (optimal: grade 6-8)
        if 6 <= grade_level <= 8:
            return 1.0
        elif 4 <= grade_level <= 10:
            return 0.8
        else:
            return 0.5

    def _analyze_content_structure(self, content: str) -> float:
        """Analyze content structure and organization."""
        scores = []
        
        # Check for main sections
        required_sections = ["introduction", "main content", "conclusion"]
        sections_found = sum(1 for section in required_sections 
                            if section.lower() in content.lower())
        scores.append(sections_found / len(required_sections))
        
        # Check for formatting elements
        formatting_elements = {
            'lists': r'<[uo]l>.*?</[uo]l>',
            'tables': r'<table>.*?</table>',
            'blockquotes': r'<blockquote>.*?</blockquote>',
            'emphasis': r'<(em|strong)>.*?</\1>'
        }
        
        formatting_score = sum(1 for pattern in formatting_elements.values()
                              if re.search(pattern, content, re.IGNORECASE | re.DOTALL))
        scores.append(formatting_score / len(formatting_elements))
        
        return sum(scores) / len(scores)

    def _analyze_technical_seo(self, content: str) -> float:
        """Analyze technical SEO elements."""
        scores = []
        
        # Check meta tags
        meta_elements = {
            'title': r'<title>.*?</title>',
            'description': r'<meta\s+name="description".*?>',
            'keywords': r'<meta\s+name="keywords".*?>',
            'viewport': r'<meta\s+name="viewport".*?>'
        }
        
        meta_score = sum(1 for pattern in meta_elements.values()
                         if re.search(pattern, content, re.IGNORECASE))
        scores.append(meta_score / len(meta_elements))
        
        # Check schema markup
        schema_score = 1.0 if 'application/ld+json' in content else 0.0
        scores.append(schema_score)
        
        # Check canonical URL
        canonical_score = 1.0 if '<link rel="canonical"' in content else 0.0
        scores.append(canonical_score)
        
        return sum(scores) / len(scores)

    def _analyze_user_experience(self, content: str) -> float:
        """Analyze user experience factors."""
        scores = []
        
        # Check for mobile responsiveness indicators
        mobile_elements = {
            'viewport': r'<meta\s+name="viewport".*?>',
            'media-queries': r'@media',
            'flexible-images': r'<img[^>]*style="[^"]*max-width:\s*100%'
        }
        
        mobile_score = sum(1 for pattern in mobile_elements.values()
                           if re.search(pattern, content, re.IGNORECASE))
        scores.append(mobile_score / len(mobile_elements))
        
        # Check for interactive elements
        interactive_elements = {
            'buttons': r'<button',
            'forms': r'<form',
            'videos': r'<video',
            'iframes': r'<iframe'
        }
        
        interactive_score = sum(1 for pattern in interactive_elements.values()
                               if pattern in content.lower())
        scores.append(interactive_score / len(interactive_elements))
        
        return sum(scores) / len(scores)

    def _analyze_content_depth(self, content: str) -> float:
        """Analyze content depth and comprehensiveness."""
        depth_indicators = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b',  # Transition words
            r'\bstudies show\b', r'\bresearch indicates\b',       # Research references
            r'\bfor example\b', r'\bcase in point\b',            # Examples
            r'\bon the other hand\b', r'\bcontrary to\b'          # Counterarguments
        ]
        
        matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                    for pattern in depth_indicators)
        return min(1.0, matches / 15)  # More than 15 depth indicators = perfect score

    def _analyze_semantic_coherence(self, content: str) -> float:
        """Analyze semantic relationships between concepts"""
        sentences = re.split(r'(?<=[.!?]) +', content)
        if len(sentences) < 2:
            return 0.0
        
        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences)-1):
            vec1 = self._sentence_vector(sentences[i])
            vec2 = self._sentence_vector(sentences[i+1])
            if vec1 and vec2:
                similarities.append(cosine_similarity([vec1], [vec2])[0][0])
        
        return sum(similarities)/len(similarities) if similarities else 0.0

    def _sentence_vector(self, sentence: str) -> List[float]:
        """Fixed-dimension sentence vectorization"""
        words = sentence.lower().split()
        # Pad/truncate to fixed 50 dimensions
        vec = [len(word) for word in words[:50]]
        return vec + [0]*(50 - len(vec))  # Pad with zeros

    def _generate_related_terms(self, keywords: List[str]) -> List[str]:
        """Generate semantically related terms for LSI."""
        related_terms = []
        for keyword in keywords:
            # Add question-based variations
            related_terms.extend([
                f"what is {keyword}",
                f"benefits of {keyword}",
                f"how to use {keyword}",
                f"{keyword} tips",
                f"best {keyword}"
            ])
            
            # Add action-oriented terms
            related_terms.extend([
                f"ultimate guide to {keyword}",
                f"{keyword} explained",
                f"mastering {keyword}",
                f"{keyword} secrets"
            ])
        return related_terms

    def _generate_seo_recommendations(self, scores: Dict) -> List[Dict]:
        """Generate specific SEO recommendations based on scores."""
        recommendations = []
        
        for factor, score in scores.items():
            if score < 0.7:
                recommendations.append({
                    "factor": factor,
                    "score": score,
                    "priority": "high" if score < 0.5 else "medium",
                    "recommendations": self._get_factor_recommendations(factor, score)
                })
        
        return recommendations

    def _get_factor_recommendations(self, factor: str, score: float) -> List[str]:
        """Get specific recommendations for improving a factor."""
        recommendations = {
            "keyword_optimization": [
                "Increase primary keyword density to 1-3%",
                "Add keywords to title and meta description",
                "Include keywords in headers and first paragraph",
                "Use keyword variations and LSI keywords"
            ],
            "content_quality": [
                "Increase content length to at least 2000 words",
                "Add more supporting media (images, videos)",
                "Include more internal and external links",
                "Break up long paragraphs"
            ],
            # Add recommendations for other factors
        }
        
        return recommendations.get(factor, ["Improve overall content quality"])

    def _count_syllables(self, word: str) -> int:
        """Count the number of syllables in a word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        
        return count

    def _generate_metadata(self, topic: str, content: str, parameters: ContentParameters) -> Dict:
        # Implementation of metadata generation
        return {}  # Placeholder, actual implementation needed

    def _check_keyword_variations(self, content: str, keyword: str) -> float:
        """Check for keyword variations and LSI keywords."""
        if not content or not keyword:
            return 0.0
        
        # Split keyword into words
        keyword_words = keyword.split()
        
        # Generate variations
        variations = []
        
        # Add singular/plural variations
        for word in keyword_words:
            if word.endswith('s'):
                variations.append(word[:-1])
            else:
                variations.append(word + 's')
        
        # Add word order variations for multi-word keywords
        if len(keyword_words) > 1:
            variations.extend([
                ' '.join(reversed(keyword_words)),
                '-'.join(keyword_words),
                ' '.join(w.capitalize() for w in keyword_words)
            ])
        
        # Add common LSI variations
        lsi_prefixes = ['best', 'top', 'how to', 'guide to', 'what is']
        lsi_suffixes = ['guide', 'tutorial', 'tips', 'review', 'comparison']
        
        variations.extend([f"{prefix} {keyword}" for prefix in lsi_prefixes])
        variations.extend([f"{keyword} {suffix}" for suffix in lsi_suffixes])
        
        # Calculate variation score
        variation_count = sum(1 for var in variations if var.lower() in content.lower())
        max_variations = len(variations)
        
        # Return normalized score (0.0 to 1.0)
        return min(1.0, variation_count / (max_variations * 0.3))  # Expect ~30% of variations

    def _find_complex_sections(self, content: str) -> List[Dict]:
        """Identify complex sections needing simplification"""
        complex_sections = []
        sentences = re.split(r'(?<=[.!?]) +', content)
        
        for i, sentence in enumerate(sentences):
            complexity_score = 0
            
            # Check for passive voice
            if re.search(r'\b(am|is|are|was|were|be|being|been)\b +[\w\s]+?ed\b', sentence):
                complexity_score += 1
                
            # Long sentence check
            if len(sentence.split()) > 25:
                complexity_score += 1
                
            # Complex word check
            complex_words = len([word for word in sentence.split() 
                                if len(word) > 7 and self.syllable_count(word) > 3])
            if complex_words > 2:
                complexity_score += 1
                
            # Jargon check
            jargon = re.findall(r'\b[A-Z]{3,}|[a-z]+-[a-z]+-?[a-z]+\b', sentence)
            if len(jargon) > 1:
                complexity_score += 1
                
            if complexity_score >= 2:
                complex_sections.append({
                    "sentence": sentence.strip(),
                    "score": complexity_score,
                    "position": i
                })
                
        return complex_sections

    def _identify_seo_opportunities(self, content: str, keywords: List[str]) -> Dict:
        """Find SEO improvement opportunities"""
        opportunities = {
            "missing_keywords": [],
            "weak_sections": [],
            "linking_opportunities": []
        }
        
        # Keyword analysis
        used_keywords = set()
        for keyword in keywords:
            if keyword.lower() not in content.lower():
                opportunities["missing_keywords"].append(keyword)
            else:
                used_keywords.add(keyword.lower())
                
        # Content weakness detection
        paragraphs = content.split('\n\n')
        for idx, para in enumerate(paragraphs):
            if len(para.split()) < 50:
                opportunities["weak_sections"].append({
                    "paragraph": idx+1,
                    "reason": "Short content",
                    "content": para[:100] + "..."
                })
                
        return opportunities

    def _detect_engagement_gaps(self, content: str) -> List[Dict]:
        """Detect sections needing engagement boosts"""
        engagement_patterns = {
            'question': r'\?',
            'statistic': r'\b\d+%?\b',
            'story': r'\b(story|example|case study)\b',
            'call_to_action': r'\b(download|learn more|sign up)\b'
        }
        
        gaps = []
        paragraphs = content.split('\n\n')
        
        for idx, para in enumerate(paragraphs):
            triggers = 0
            for pattern in engagement_patterns.values():
                if re.search(pattern, para, re.IGNORECASE):
                    triggers += 1
                    
            if triggers < 2:
                gaps.append({
                    "paragraph": idx+1,
                    "score": triggers,
                    "content": para[:150] + "..."
                })
                
        return gaps

    def syllable_count(self, word: str) -> int:
        """Approximate syllable count for English words"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count

    def _create_base_prompt(self, topic: str, keywords: List[str], parameters: ContentParameters) -> str:
        """Create initial generation prompt using content type template"""
        return CONTENT_TYPES[parameters.content_type].prompt_template.format(
            topic=topic,
            audience=parameters.target_audience,
            tone=parameters.tone,
            language=parameters.language,
            word_count=parameters.word_count,
            keywords=', '.join(keywords)
        )

    def _create_refinement_prompt(self, content: str, analysis: Dict, parameters: ContentParameters) -> str:
        """Create content refinement prompt with analysis data"""
        return f"Refine this content:\n{content}\n\nAnalysis:\n{json.dumps(analysis)}\n\nImprove while maintaining {parameters.tone} tone."

    def _create_verification_prompt(self, content: str, verification: Dict) -> str:
        """Create content verification prompt"""
        return f"Verify and correct:\n{content}\n\nIssues found:\n{json.dumps(verification)}"

    def _create_engagement_prompt(self, content: str, parameters: ContentParameters) -> str:
        """Create engagement optimization prompt"""
        return f"Optimize for engagement ({parameters.target_audience}):\n{content}"

    def _check_fact_consistency(self, content: str) -> bool:
        """Verify factual consistency throughout content"""
        # Simple implementation - check for conflicting statements
        claims = re.findall(r'(?:[A-Z][^\.!?]*[\.!?])', content)
        unique_claims = set(claims)
        return len(claims) == len(unique_claims)  # No duplicates

    def _analyze_argument_flow(self, content: str) -> float:
        """Analyze logical flow between paragraphs"""
        transitions = ['however', 'moreover', 'therefore', 'consequently']
        transition_count = sum(content.lower().count(t) for t in transitions)
        return min(1.0, transition_count / 5)  # 5+ transitions = perfect

    def _verify_conceptual_accuracy(self, content: str) -> bool:
        """Basic conceptual accuracy check"""
        # Placeholder - integrate with knowledge base in real implementation
        common_errors = ['their are', 'your welcome', 'should of']
        return not any(error in content.lower() for error in common_errors)

    def _check_style_consistency(self, content: str) -> float:
        """Check for consistent style/tone"""
        tone_indicators = {
            'professional': ['furthermore', 'moreover', 'thus'],
            'conversational': ['you\'ll', 'we\'ve', 'let\'s']
        }
        matches = {tone: sum(content.count(word) for word in words) 
                  for tone, words in tone_indicators.items()}
        return max(matches.values()) / (sum(matches.values()) + 1)

    def _verify_keyword_usage(self, content: str, keywords: List[str]) -> bool:
        """Verify proper keyword implementation"""
        if not keywords:
            return True
        
        content_lower = content.lower()
        return all(
            keyword.lower() in content_lower and 
            0.01 <= (content_lower.count(keyword.lower())/len(content.split())) <= 0.03
            for keyword in keywords
        ) and self._check_keyword_variations(content, keywords[0]) >= 0.5

    def _verify_content_depth(self, content: str) -> bool:
        """Verify content meets depth requirements"""
        depth_indicators = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bstudies show\b', r'\bresearch indicates\b',
            r'\bfor example\b', r'\bcase in point\b',
            r'\bon the other hand\b', r'\bcontrary to\b'
        ]
        return sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                 for pattern in depth_indicators) >= 5

REFINEMENT_PROMPT = """
Analyze this content draft and suggest improvements considering:
- Current SEO score: {seo_score}
- Identified issues: {issues}

Apply these refinement strategies:
1. Simplify complex sections using analogies and examples.
2. Strengthen weak arguments with supporting evidence.
3. Improve flow between sections.
4. Enhance keyword integration naturally.
5. Add a dedicated benchmark comparison section.
6. Use language that powerfully conveys the user experience.
7. Conclude with a clear and persuasive call to action.

Provide the refined version maintaining the core message but improving:
- Readability (target grade {readability_grade})
- Engagement metrics
- SEO effectiveness
"""

VERIFICATION_PROMPT = """
Verify the accuracy of this content by:
1. Checking factual consistency
2. Validating source citations
3. Ensuring logical argument flow
4. Confirming technical terminology accuracy
5. Maintaining consistent tone ({tone})

List any discrepancies found and provide corrected versions for problematic sections.
"""

SIMPLIFICATION_PROMPT = """
Simplify this content for {audience} audience:
- Break down complex concepts using:
  - Real-world analogies
  - Step-by-step explanations
  - Visual metaphors
- Ensure technical accuracy while improving accessibility
- Maintain original SEO structure and keywords
- Target readability level: grade {readability}
"""

async def main():
    # Initialize the generator
    config = Config(api_key="your_hugging_face_api_key")
    seo_params = SEOParams(min_word_count=300)
    generator = SEOContentGenerator(config, seo_params)
    
    # Generate content
    result = await generator.generate_seo_content(
        topic="AI in Healthcare",
        keywords=["artificial intelligence healthcare", "medical AI", "AI diagnosis"],
        parameters=ContentParameters(content_type="article", target_audience="general", tone="professional", language="en", word_count=300),
        options={"competitors": ["competitor1.com", "competitor2.com"]}
    )
    
    print(result["content"])
    print(result["seo_score"])