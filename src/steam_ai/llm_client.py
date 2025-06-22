import logging
from typing import List, Dict, Any, Optional, Generator
import time
from dataclasses import dataclass
from enum import Enum

from .config import Config
from .http_client import SyncHTTPClient

logger = logging.getLogger(__name__)


class PromptTemplate(Enum):
    """Predefined prompt templates for different use cases."""
    BASIC_QA = "basic_qa"
    DETAILED_ANALYSIS = "detailed_analysis"
    SENTIMENT_SUMMARY = "sentiment_summary"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"


@dataclass
class LLMResponse:
    """Container for LLM response with metadata."""
    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    processing_time: float = 0.0
    temperature: float = 0.0
    cached: bool = False


@dataclass
class PromptContext:
    """Context for prompt generation."""
    reviews: List[str]
    question: str
    app_name: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None


class PromptManager:
    """Manages prompt templates and generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        # Get language setting for templates
        language = self.config.app_language.lower()
        
        if language == "pl":
            return self._get_polish_templates()
        else:
            return self._get_english_templates()
    
    def _get_polish_templates(self) -> Dict[str, str]:
        """Polish prompt templates."""
        return {
            PromptTemplate.BASIC_QA.value: """
Na podstawie poniższych recenzji gry odpowiedz na pytanie użytkownika zwięźle i dokładnie. ODPOWIADAJ ZAWSZE PO POLSKU.

Recenzje:
{context}

Pytanie: {question}

Instrukcje:
- Skup się na najważniejszych informacjach z recenzji
- Podaj zrównoważoną perspektywę uwzględniającą opinie pozytywne i negatywne
- Odpowiedź powinna być zwięzła ale informacyjna
- Jeśli recenzje nie zawierają wystarczających informacji, napisz o tym
- ODPOWIADAJ TYLKO PO POLSKU

Odpowiedź:""",

            PromptTemplate.DETAILED_ANALYSIS.value: """
Przeanalizuj poniższe recenzje gry, aby udzielić wyczerpującej odpowiedzi na pytanie użytkownika. ODPOWIADAJ ZAWSZE PO POLSKU.

Gra: {app_name}
Recenzje (łącznie {review_count}):
{context}

Pytanie: {question}

Instrukcje:
- Podaj szczegółową analizę opartą na treści recenzji
- Uwzględnij konkretne przykłady z recenzji, gdy są istotne
- Rozważ różne perspektywy graczy i style gry
- Wspomnij o wspólnych tematach lub powtarzających się problemach
- Uporządkuj odpowiedź w jasne punkty główne
- ODPOWIADAJ TYLKO PO POLSKU

Analiza:""",

            PromptTemplate.SENTIMENT_SUMMARY.value: """
Podsumuj ogólne nastroje i kluczowe tematy z tych recenzji gry. ODPOWIADAJ ZAWSZE PO POLSKU.

Gra: {app_name}
Recenzje:
{context}

Pytanie: {question}

Instrukcje:
- Zidentyfikuj główne tematy pozytywne i negatywne
- Podaj ogólną ocenę nastrojów
- Wyróżnij najczęściej wspominane problemy lub pochwały
- Rozważ intensywność wyrażanych opinii
- Daj zrównoważony przegląd, który pomoże potencjalnym nabywcom
- ODPOWIADAJ TYLKO PO POLSKU

Podsumowanie:""",

            PromptTemplate.COMPARISON.value: """
Porównaj różne aspekty tej gry na podstawie dostarczonych recenzji użytkowników. ODPOWIADAJ ZAWSZE PO POLSKU.

Recenzje:
{context}

Pytanie: {question}

Instrukcje:
- Porównaj różne punkty widzenia przedstawione w recenzjach
- Podkreśl obszary zgodności i niezgodności
- Zidentyfikuj czynniki, które mogą wpływać na różne opinie (czas gry, typ gracza, itp.)
- Podaj zrównoważone porównanie, które uwzględnia wiele perspektyw
- ODPOWIADAJ TYLKO PO POLSKU

Porównanie:""",

            PromptTemplate.RECOMMENDATION.value: """
Na podstawie tych recenzji gry podaj rekomendację, która odpowiada na pytanie użytkownika. ODPOWIADAJ ZAWSZE PO POLSKU.

Recenzje:
{context}

Pytanie: {question}

Instrukcje:
- Rozważ, do kogo ta gra może przemawiać na podstawie recenzji
- Wspomnij o wszelkich ważnych zastrzeżeniach lub rozważaniach
- Bądź szczery co do potencjalnych wad wymienionych w recenzjach
- Podaj praktyczne porady dla użytkownika
- Rozważ różne preferencje graczy i sytuacje
- ODPOWIADAJ TYLKO PO POLSKU

Rekomendacja:"""
        }
    
    def _get_english_templates(self) -> Dict[str, str]:
        """English prompt templates."""
        return {
            PromptTemplate.BASIC_QA.value: """
Based on the following game reviews, answer the user's question concisely and accurately. ALWAYS RESPOND IN ENGLISH.

Reviews:
{context}

Question: {question}

Instructions:
- Focus on the most relevant information from the reviews
- Provide a balanced perspective considering both positive and negative feedback
- Keep your answer concise but informative
- If the reviews don't contain enough information to answer the question, say so
- RESPOND ONLY IN ENGLISH

Answer:""",

            PromptTemplate.DETAILED_ANALYSIS.value: """
Analyze the following game reviews to provide a comprehensive answer to the user's question. ALWAYS RESPOND IN ENGLISH.

Game: {app_name}
Reviews ({review_count} total):
{context}

Question: {question}

Instructions:
- Provide a detailed analysis based on the review content
- Include specific examples from the reviews when relevant
- Consider different player perspectives and playstyles
- Mention any common themes or recurring issues
- Structure your response clearly with main points
- RESPOND ONLY IN ENGLISH

Analysis:""",

            PromptTemplate.SENTIMENT_SUMMARY.value: """
Summarize the overall sentiment and key themes from these game reviews. ALWAYS RESPOND IN ENGLISH.

Game: {app_name}
Reviews:
{context}

Question: {question}

Instructions:
- Identify the main positive and negative themes
- Provide an overall sentiment assessment
- Highlight the most frequently mentioned issues or praise
- Consider the intensity of opinions expressed
- Give a balanced overview that helps potential buyers
- RESPOND ONLY IN ENGLISH

Summary:""",

            PromptTemplate.COMPARISON.value: """
Compare different aspects of this game based on the user reviews provided. ALWAYS RESPOND IN ENGLISH.

Reviews:
{context}

Question: {question}

Instructions:
- Compare different viewpoints presented in the reviews
- Highlight areas of consensus and disagreement
- Identify factors that might influence different opinions (playtime, player type, etc.)
- Provide a balanced comparison that acknowledges multiple perspectives
- RESPOND ONLY IN ENGLISH

Comparison:""",

            PromptTemplate.RECOMMENDATION.value: """
Based on these game reviews, provide a recommendation that addresses the user's question. ALWAYS RESPOND IN ENGLISH.

Reviews:
{context}

Question: {question}

Instructions:
- Consider who this game would appeal to based on the reviews
- Mention any important caveats or considerations
- Be honest about potential drawbacks mentioned in reviews
- Provide actionable advice for the user
- Consider different player preferences and situations
- RESPOND ONLY IN ENGLISH

Recommendation:"""
        }
    
    def generate_prompt(self, 
                       context: PromptContext, 
                       template: PromptTemplate = PromptTemplate.BASIC_QA,
                       max_context_length: int = 8000) -> str:
        """
        Generate a prompt from context and template.
        
        Args:
            context: Prompt context with reviews and question
            template: Template to use
            max_context_length: Maximum length for context section
            
        Returns:
            str: Generated prompt
        """
        # Prepare context text
        context_text = self._prepare_context_text(context.reviews, max_context_length)
        
        # Get template
        template_str = self.templates.get(template.value, self.templates[PromptTemplate.BASIC_QA.value])
        
        # Format template
        format_kwargs = {
            "context": context_text,
            "question": context.question,
            "review_count": len(context.reviews),
            "app_name": context.app_name or "Unknown Game"
        }
        
        # Add additional context if provided
        if context.additional_context:
            format_kwargs.update(context.additional_context)
        
        return template_str.format(**format_kwargs)
    
    def _prepare_context_text(self, reviews: List[str], max_length: int) -> str:
        """
        Prepare context text from reviews with length management.
        
        Args:
            reviews: List of review texts
            max_length: Maximum total length
            
        Returns:
            str: Prepared context text
        """
        if not reviews:
            return "No reviews available."
        
        # Start with all reviews
        context_parts = []
        current_length = 0
        
        for i, review in enumerate(reviews):
            review_text = f"Review {i+1}: {review}\n\n"
            
            if current_length + len(review_text) <= max_length:
                context_parts.append(review_text)
                current_length += len(review_text)
            else:
                # Try to fit a truncated version
                remaining_space = max_length - current_length - 50  # Leave space for truncation message
                if remaining_space > 100:  # Only truncate if we have reasonable space
                    truncated_review = review[:remaining_space] + "..."
                    context_parts.append(f"Review {i+1}: {truncated_review}\n\n")
                break
        
        if not context_parts:
            # If no reviews fit, truncate the first one
            first_review = reviews[0][:max_length-50] + "..."
            return f"Review 1: {first_review}"
        
        return "".join(context_parts).strip()
    
    def suggest_template(self, question: str) -> PromptTemplate:
        """
        Suggest the best template based on the question content.
        
        Args:
            question: User's question
            
        Returns:
            PromptTemplate: Suggested template
        """
        question_lower = question.lower()
        
        # Keywords for different templates
        analysis_keywords = ["analyze", "analysis", "detailed", "comprehensive", "deep dive"]
        sentiment_keywords = ["sentiment", "opinion", "feel", "think", "overall", "general"]
        comparison_keywords = ["compare", "comparison", "versus", "vs", "difference", "better"]
        recommendation_keywords = ["recommend", "should i", "worth it", "buy", "purchase"]
        
        if any(keyword in question_lower for keyword in recommendation_keywords):
            return PromptTemplate.RECOMMENDATION
        elif any(keyword in question_lower for keyword in comparison_keywords):
            return PromptTemplate.COMPARISON
        elif any(keyword in question_lower for keyword in sentiment_keywords):
            return PromptTemplate.SENTIMENT_SUMMARY
        elif any(keyword in question_lower for keyword in analysis_keywords):
            return PromptTemplate.DETAILED_ANALYSIS
        else:
            return PromptTemplate.BASIC_QA


class LLMClient:
    """Optimized LLM client with caching and resource management."""
    
    def __init__(self, config: Config):
        self.config = config
        self.http_client = SyncHTTPClient(
            pool_size=config.connection_pool_size,
            timeout=config.ollama_timeout_llm
        )
        self.prompt_manager = PromptManager(config)
        self._response_cache: Dict[str, LLMResponse] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def generate_response(self, 
                         context: PromptContext,
                         template: Optional[PromptTemplate] = None,
                         temperature: float = 0.1,
                         max_tokens: Optional[int] = None,
                         use_cache: bool = True) -> LLMResponse:
        """
        Generate response from LLM with caching and optimization.
        
        Args:
            context: Prompt context
            template: Template to use (auto-selected if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use response caching
            
        Returns:
            LLMResponse: Generated response
        """
        # Auto-select template if not provided
        if template is None:
            template = self.prompt_manager.suggest_template(context.question)
        
        # Generate prompt
        prompt = self.prompt_manager.generate_prompt(context, template)
        
        # Check cache
        cache_key = self._get_cache_key(prompt, temperature, max_tokens)
        if use_cache and cache_key in self._response_cache:
            self._cache_hits += 1
            cached_response = self._response_cache[cache_key]
            cached_response.cached = True
            return cached_response
        
        self._cache_misses += 1
        
        # Generate new response
        start_time = time.time()
        response_text = self._call_llm(prompt, temperature, max_tokens)
        processing_time = time.time() - start_time
        
        # Create response object
        response = LLMResponse(
            text=response_text,
            model=self.config.llm_model,
            processing_time=processing_time,
            temperature=temperature,
            cached=False
        )
        
        # Cache response
        if use_cache:
            self._response_cache[cache_key] = response
        
        return response
    
    def _call_llm(self, 
                  prompt: str, 
                  temperature: float = 0.1,
                  max_tokens: Optional[int] = None) -> str:
        """
        Make actual LLM API call.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated response text
        """
        payload = {
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or 500,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = self.http_client.post(
                self.config.ollama_llm_url,
                json_data=payload
            )
            
            response_text = response.get("response", "")
            if not response_text:
                logger.warning("Empty response from LLM")
                return "I apologize, but I couldn't generate a response. Please try again."
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _get_cache_key(self, prompt: str, temperature: float, max_tokens: Optional[int]) -> str:
        """Generate cache key for prompt and parameters."""
        import hashlib
        
        key_data = f"{prompt}_{temperature}_{max_tokens}_{self.config.llm_model}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def generate_streaming_response(self, 
                                   context: PromptContext,
                                   template: Optional[PromptTemplate] = None,
                                   temperature: float = 0.1) -> Generator[str, None, None]:
        """
        Generate streaming response (placeholder for future implementation).
        
        Args:
            context: Prompt context
            template: Template to use
            temperature: Sampling temperature
            
        Yields:
            str: Response chunks
        """
        # This would be implemented for streaming responses
        # For now, return the full response
        response = self.generate_response(context, template, temperature, use_cache=False)
        yield response.text
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._response_cache)
        }
    
    def optimize_cache(self, max_cache_size: int = 100) -> None:
        """Optimize cache by removing oldest entries."""
        if len(self._response_cache) > max_cache_size:
            # Simple FIFO eviction (could be improved with LRU)
            items = list(self._response_cache.items())
            self._response_cache = dict(items[-max_cache_size//2:])
    
    def validate_model_availability(self) -> bool:
        """
        Check if the configured LLM model is available.
        
        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            # Simple test call
            test_payload = {
                "model": self.config.llm_model,
                "prompt": "Test",
                "stream": False,
                "options": {"num_predict": 1}
            }
            
            response = self.http_client.post(
                self.config.ollama_llm_url,
                json_data=test_payload
            )
            
            return "response" in response
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the configured model.
        
        Returns:
            Optional[Dict[str, Any]]: Model information or None if unavailable
        """
        try:
            # This would call Ollama's model info endpoint
            # For now, return basic info
            return {
                "model": self.config.llm_model,
                "available": self.validate_model_availability(),
                "timeout": self.config.ollama_timeout_llm
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None


class ConversationManager:
    """Manages conversation context and history."""
    
    def __init__(self, config: Config):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10
    
    def add_exchange(self, question: str, answer: str) -> None:
        """Add question-answer exchange to history."""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_context_aware_prompt(self, 
                                context: PromptContext,
                                include_history: bool = True) -> PromptContext:
        """
        Create context-aware prompt that includes conversation history.
        
        Args:
            context: Current prompt context
            include_history: Whether to include conversation history
            
        Returns:
            PromptContext: Enhanced context with history
        """
        if not include_history or not self.conversation_history:
            return context
        
        # Add conversation history to additional context
        history_text = self._format_conversation_history()
        
        additional_context = context.additional_context or {}
        additional_context["conversation_history"] = history_text
        
        return PromptContext(
            reviews=context.reviews,
            question=context.question,
            app_name=context.app_name,
            additional_context=additional_context
        )
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for inclusion in prompts."""
        if not self.conversation_history:
            return ""
        
        history_parts = ["Previous conversation:"]
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            history_parts.append(f"Q: {exchange['question']}")
            answer_text = str(exchange['answer'])[:200] + "..."  # Truncate long answers
            history_parts.append(f"A: {answer_text}")
        
        return "\n".join(history_parts)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history."""
        return {
            "total_exchanges": len(self.conversation_history),
            "latest_timestamp": self.conversation_history[-1]["timestamp"] if self.conversation_history else None,
            "max_length": self.max_history_length
        }