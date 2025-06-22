import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


class TranslationManager:
    """Manages translations and internationalization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.language = config.app_language
        self.i18n_dir = Path(config.i18n_dir)
        self.translations: Dict[str, str] = {}
        self.fallback_translations: Dict[str, str] = {}
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self) -> None:
        """Load translations for the configured language."""
        try:
            # Load primary language
            primary_path = self.i18n_dir / f"{self.language}.json"
            if primary_path.exists():
                with open(primary_path, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
                logger.info(f"Loaded translations for language: {self.language}")
            else:
                logger.warning(f"Translation file not found: {primary_path}")
            
            # Load fallback language (English or Polish)
            fallback_lang = "en" if self.language != "en" else "pl"
            fallback_path = self.i18n_dir / f"{fallback_lang}.json"
            if fallback_path.exists():
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    self.fallback_translations = json.load(f)
                logger.info(f"Loaded fallback translations: {fallback_lang}")
            
        except Exception as e:
            logger.error(f"Error loading translations: {e}")
            # Use minimal fallback translations
            self._load_minimal_fallback()
    
    def _load_minimal_fallback(self) -> None:
        """Load minimal fallback translations in case of error."""
        if self.language == "pl":
            self.translations = {
                "welcome": "🎮 Witaj w analizatorze recenzji Steam z RAG!",
                "ask_app_id": "🆔 Podaj Steam App ID gry (np. 578080 dla PUBG): ",
                "use_saved": "🔄 Znaleziono zapisane dane dla {app_id}. Użyć je? (t/n): ",
                "session_loaded": "✅ Sesja załadowana z {count} recenzjami",
                "session_ready": "✅ Sesja gotowa! Możesz teraz zadawać pytania.",
                "ready": "✅ Gotowe! Możesz teraz zadawać pytania na podstawie recenzji.",
                "exit_hint": "Wpisz 'exit' lub 'q' aby zakończyć.\n",
                "your_question": "> Twoje pytanie: ",
                "goodbye": "👋 Do zobaczenia!",
                "answer": "\n📋 Odpowiedź:\n",
                "generating": "Generowanie odpowiedzi",
                "searching": "Przeszukiwanie odpowiednich recenzji",
                "yes": "t",
                "no": "n",
                "exit_cmd": "exit",
                "interrupted": "Operacja przerwana",
                "app_not_found": "Nie można zweryfikować informacji o grze, kontynuujemy mimo to",
                "session_setup_error": "Błąd konfiguracji sesji: {error}",
                "session_load_error": "Błąd ładowania sesji: {error}",
                "session_create_error": "Błąd tworzenia sesji: {error}",
                "creating_session": "Tworzenie nowej sesji...",
                "no_reviews_found": "Nie znaleziono recenzji dla tej gry",
                "session_error": "Błąd sesji - proszę przeładować",
                "no_relevant_reviews": "Nie znaleziono odpowiednich recenzji dla Twojego pytania",
                "processing_error": "Błąd przetwarzania pytania: {error}",
                "question_error": "Błąd z pytaniem: {error}",
                "unexpected_error": "Nieoczekiwany błąd: {error}",
                "continue_after_error": "Czy chcesz kontynuować?",
                "llm_prompt": "Odpowiedz na pytanie na podstawie recenzji gry. Odpowiadaj ZAWSZE PO POLSKU:\n\nRecenzje:\n{context}\n\nPytanie: {question}\n\nInstrukcje:\n- Odpowiedz w języku polskim\n- Skup się na najważniejszych informacjach z recenzji\n- Podaj zrównoważoną perspektywę uwzględniającą opinie pozytywne i negatywne\n- Jeśli recenzje nie zawierają wystarczających informacji, napisz o tym\n\nOdpowiedź:"
            }
        else:
            self.translations = {
                "welcome": "🎮 Welcome to the Steam review analyzer with RAG!",
                "ask_app_id": "🆔 Enter the game's Steam App ID (e.g. 578080 for PUBG): ",
                "use_saved": "🔄 Found saved data for {app_id}. Use it? (y/n): ",
                "session_loaded": "✅ Session loaded with {count} reviews",
                "session_ready": "✅ Session ready! You can now ask questions.",
                "ready": "✅ Ready! You can now ask questions based on the reviews.",
                "exit_hint": "Type 'exit' or 'q' to quit.\n",
                "your_question": "> Your question: ",
                "goodbye": "👋 Goodbye!",
                "answer": "\n📋 Answer:\n",
                "generating": "Generating answer",
                "searching": "Searching relevant reviews",
                "yes": "y",
                "no": "n",
                "exit_cmd": "exit",
                "interrupted": "Operation interrupted",
                "app_not_found": "Could not verify app information, proceeding anyway",
                "session_setup_error": "Error setting up session: {error}",
                "session_load_error": "Error loading session: {error}",
                "session_create_error": "Error creating session: {error}",
                "creating_session": "Creating new session...",
                "no_reviews_found": "No reviews found for this app",
                "session_error": "Session error - please reload",
                "no_relevant_reviews": "No relevant reviews found for your question",
                "processing_error": "Error processing question: {error}",
                "question_error": "Error with question: {error}",
                "unexpected_error": "Unexpected error: {error}",
                "continue_after_error": "Would you like to continue?",
                "llm_prompt": "Answer the question based on the game reviews. ALWAYS RESPOND IN ENGLISH:\n\nReviews:\n{context}\n\nQuestion: {question}\n\nInstructions:\n- Focus on the most relevant information from the reviews\n- Provide a balanced perspective considering both positive and negative feedback\n- Keep your answer concise but informative\n- If the reviews don't contain enough information to answer the question, say so\n- RESPOND ONLY IN ENGLISH\n\nAnswer:"
            }
        self.fallback_translations = self.translations.copy()
    
    def tr(self, key: str, **kwargs) -> str:
        """
        Translate a key with optional formatting parameters.
        
        Args:
            key: Translation key
            **kwargs: Formatting parameters
            
        Returns:
            str: Translated and formatted text
        """
        # Try primary language first
        template = self.translations.get(key)
        
        # Fall back to fallback language
        if template is None:
            template = self.fallback_translations.get(key)
        
        # Fall back to key itself
        if template is None:
            logger.warning(f"Translation not found for key: {key}")
            template = key
        
        # Format with parameters if provided
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Error formatting translation '{key}': {e}")
                return template
        
        return template
    
    def add_translation(self, key: str, value: str, language: Optional[str] = None) -> None:
        """
        Add or update a translation.
        
        Args:
            key: Translation key
            value: Translation value
            language: Language code (defaults to current language)
        """
        if language is None or language == self.language:
            self.translations[key] = value
        elif language == "en" or language == "pl":
            self.fallback_translations[key] = value
    
    def get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        languages = []
        if self.i18n_dir.exists():
            for file_path in self.i18n_dir.glob("*.json"):
                languages.append(file_path.stem)
        return sorted(languages)
    
    def switch_language(self, language: str) -> bool:
        """
        Switch to a different language.
        
        Args:
            language: Language code to switch to
            
        Returns:
            bool: True if switch was successful, False otherwise
        """
        if language == self.language:
            return True
        
        # Check if language is available
        language_path = self.i18n_dir / f"{language}.json"
        if not language_path.exists():
            logger.warning(f"Language file not found: {language}")
            return False
        
        try:
            # Load new language
            with open(language_path, 'r', encoding='utf-8') as f:
                new_translations = json.load(f)
            
            # Update current language
            self.language = language
            self.translations = new_translations
            
            logger.info(f"Switched to language: {language}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to language {language}: {e}")
            return False
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded translations."""
        return {
            "current_language": self.language,
            "primary_translations": len(self.translations),
            "fallback_translations": len(self.fallback_translations),
            "available_languages": self.get_available_languages(),
            "i18n_directory": str(self.i18n_dir)
        }
    
    def validate_translations(self) -> Dict[str, list[str]]:
        """
        Validate translations for missing keys or formatting issues.
        
        Returns:
            Dict[str, list[str]]: Validation results with issues found
        """
        issues = {
            "missing_keys": [],
            "formatting_errors": [],
            "unused_keys": []
        }
        
        # Define required keys
        required_keys = {
            "welcome", "ask_app_id", "use_saved", "session_loaded", "session_ready",
            "ready", "exit_hint", "your_question", "goodbye", "answer", "generating",
            "searching", "yes", "no", "exit_cmd"
        }
        
        # Check for missing required keys
        for key in required_keys:
            if key not in self.translations and key not in self.fallback_translations:
                issues["missing_keys"].append(key)
        
        # Check for formatting issues
        for key, template in self.translations.items():
            if "{" in template and "}" in template:
                try:
                    # Try formatting with dummy parameters
                    dummy_params = {
                        "app_id": "123456",
                        "count": "100",
                        "error": "test error"
                    }
                    template.format(**dummy_params)
                except (KeyError, ValueError) as e:
                    issues["formatting_errors"].append(f"{key}: {str(e)}")
        
        return issues
    
    def export_translations(self, output_path: str) -> bool:
        """
        Export current translations to a file.
        
        Args:
            output_path: Path to save translations
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.translations, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Translations exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting translations: {e}")
            return False