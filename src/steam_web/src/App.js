import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css';

// Markdown configuration
const markdownComponents = {
  // Custom link component for security
  a: ({ href, children, ...props }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
      {children}
    </a>
  ),
  // Custom code block styling
  pre: ({ children, ...props }) => (
    <pre className="markdown-code-block" {...props}>
      {children}
    </pre>
  ),
  // Custom inline code styling
  code: ({ children, className, ...props }) => (
    <code className={`markdown-code ${className || ''}`} {...props}>
      {children}
    </code>
  ),
  // Custom paragraph styling
  p: ({ children, ...props }) => (
    <p className="markdown-paragraph" {...props}>
      {children}
    </p>
  ),
  // Custom list styling
  ul: ({ children, ...props }) => (
    <ul className="markdown-list" {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }) => (
    <ol className="markdown-list" {...props}>
      {children}
    </ol>
  ),
  // Custom heading styling
  h1: ({ children, ...props }) => <h1 className="markdown-heading" {...props}>{children}</h1>,
  h2: ({ children, ...props }) => <h2 className="markdown-heading" {...props}>{children}</h2>,
  h3: ({ children, ...props }) => <h3 className="markdown-heading" {...props}>{children}</h3>,
  h4: ({ children, ...props }) => <h4 className="markdown-heading" {...props}>{children}</h4>,
  h5: ({ children, ...props }) => <h5 className="markdown-heading" {...props}>{children}</h5>,
  h6: ({ children, ...props }) => <h6 className="markdown-heading" {...props}>{children}</h6>,
};

const FUNNY_LOADING_MESSAGES = [
  "🔍 Scanning Steam reviews for hidden insights...",
  "🎮 Asking the review gods for wisdom...",
  "🤖 Teaching AI to understand gamer language...",
  "📊 Counting review thumbs up and down...",
  "🎯 Analyzing what players really think...",
  "🔥 Warming up the sentiment analysis engine...",
  "🎲 Rolling dice to find the most helpful reviews...",
  "📈 Processing years of player feedback...",
  "⚡ Channeling the power of community opinions...",
  "🎪 Juggling thousands of honest reviews...",
  "🔮 Consulting the crystal ball of player sentiment...",
  "🎨 Painting a picture of player experiences...",
  "🚀 Launching review analysis rockets...",
  "🎵 Listening to the symphony of player voices...",
  "🏗️ Building bridges between reviews and insights...",
  "🌟 Mining review gold from Steam data...",
  "🎭 Performing interpretive dance with review text...",
  "🔬 Scientifically analyzing player opinions...",
  "🎈 Inflating the review processing algorithm...",
  "⚙️ Calibrating the honesty detection sensors..."
];



const TRANSLATIONS = {
  en: {
    title: "SteamLens AI",
    subtitle: "See through the fog of hype with AI-powered review analysis.",
    startOver: "Start Over",
    searchPlaceholder: "Enter a game name to search (e.g., 'Cyberpunk 2077', 'The Witcher 3')...",
    chatPlaceholder: "Ask me anything about the game reviews...",
    welcomeMessage: "Hi! I'm **SteamLens AI**, your Steam review analyzer. I help you understand what players *really* think about games by analyzing thousands of Steam reviews.\n\n**Let's start by searching for a game you're interested in!**",
    foundGames: "Found {count} games matching \"{query}\". Click on a game below to analyze its reviews:",
    noGamesFound: "Sorry, I couldn't find any games matching \"{query}\". Try a different search term or check the spelling.",
    gameSelected: "Great choice! You selected \"{name}\". I'm now initializing the analysis session and downloading Steam reviews. This might take a few minutes...",
    existingSession: "Perfect! I found an existing analysis with {count} reviews for \"{name}\". You can now ask me questions about what players think about this game!",
    sessionReady: "Perfect! I've analyzed **{count} reviews** for *\"{name}\"*. You can now ask me questions about what players think about this game.\n\n## Try asking things like:\n\n• **\"Is this game worth buying?\"**\n• **\"What are the main complaints?\"**\n• **\"How's the performance and bugs?\"**\n• **\"What do players love most about it?\"**",
    sessionError: "Sorry, there was an error processing the reviews for \"{name}\". {error} Please try again later.",
    restartMessage: "Let's start over! What game would you like me to analyze?",
    searchError: "Sorry, I encountered an error while searching for games. Please make sure the API server is running and try again.",
    initError: "Sorry, I couldn't initialize the analysis session for \"{name}\". Error: {error}. Please try again later.",
    processingTimeout: "Processing is taking longer than expected. The session might still be working in the background. Please try starting over or check back later.",
    sessionNotFound: "Session not found. This might happen if the session expired or there was a server restart. Please try starting over.",
    serverError: "The server encountered an error while processing reviews. Please try starting over.",
    connectionLost: "Lost connection while processing reviews after multiple attempts. Please check if the API server is running and try starting over.",
    questionError: "Sorry, I encountered an error while analyzing the reviews for your question.",
    tooManyRequests: "Too many requests. Please wait a moment before asking another question.",
    noAnswer: "No answer received from server"
  },
  pl: {
    title: "SteamLens AI",
    subtitle: "Przejrzyj przez mgłę hype’u dzięki analizie recenzji opartej na AI",
    startOver: "Zacznij Od Nowa",
    searchPlaceholder: "Wpisz nazwę gry do wyszukania (np. 'Cyberpunk 2077', 'The Witcher 3')...",
    chatPlaceholder: "Zapytaj mnie o cokolwiek dotyczące recenzji gry...",
    welcomeMessage: "Cześć! Jestem **SteamLens AI**, analizatorem recenzji Steam. Pomagam zrozumieć, co *naprawdę* myślą gracze o grach, analizując tysiące recenzji Steam.\n\n**Zacznijmy od wyszukania gry, która Cię interesuje!**",
    foundGames: "Znaleziono {count} gier pasujących do \"{query}\". Kliknij na grę poniżej, aby przeanalizować jej recenzje:",
    noGamesFound: "Przepraszam, nie mogłem znaleźć żadnych gier pasujących do \"{query}\". Spróbuj innego hasła wyszukiwania lub sprawdź pisownię.",
    gameSelected: "Świetny wybór! Wybrałeś \"{name}\". Teraz inicjalizuję sesję analizy i pobieram recenzje Steam. To może potrwać kilka minut...",
    existingSession: "Świetnie! Znalazłem istniejącą analizę z {count} recenzjami dla \"{name}\". Możesz teraz zadawać mi pytania o to, co myślą gracze o tej grze!",
    sessionReady: "Świetnie! Przeanalizowałem **{count} recenzji** dla *\"{name}\"*. Możesz teraz zadawać mi pytania o to, co myślą gracze o tej grze.\n\n## Spróbuj zapytać o rzeczy takie jak:\n\n• **\"Czy ta gra jest warta kupienia?\"**\n• **\"Jakie są główne skargi?\"**\n• **\"Jak wygląda wydajność i błędy?\"**\n• **\"Co gracze najbardziej kochają w tej grze?\"**",
    sessionError: "Przepraszam, wystąpił błąd podczas przetwarzania recenzji dla \"{name}\". {error} Spróbuj ponownie później.",
    restartMessage: "Zacznijmy od nowa! Którą grę chciałbyś przeanalizować?",
    searchError: "Przepraszam, napotkałem błąd podczas wyszukiwania gier. Upewnij się, że serwer API działa i spróbuj ponownie.",
    initError: "Przepraszam, nie mogłem zainicjalizować sesji analizy dla \"{name}\". Błąd: {error}. Spróbuj ponownie później.",
    processingTimeout: "Przetwarzanie trwa dłużej niż oczekiwano. Sesja może nadal działać w tle. Spróbuj zacząć od nowa lub sprawdź później.",
    sessionNotFound: "Sesja nie została znaleziona. Może się to zdarzyć, jeśli sesja wygasła lub nastąpił restart serwera. Spróbuj zacząć od nowa.",
    serverError: "Serwer napotkał błąd podczas przetwarzania recenzji. Spróbuj zacząć od nowa.",
    connectionLost: "Utracono połączenie podczas przetwarzania recenzji po wielu próbach. Sprawdź, czy serwer API działa i spróbuj zacząć od nowa.",
    questionError: "Przepraszam, napotkałem błąd podczas analizowania recenzji dla Twojego pytania.",
    tooManyRequests: "Za dużo zapytań. Poczekaj chwilę przed zadaniem kolejnego pytania.",
    noAnswer: "Nie otrzymano odpowiedzi z serwera"
  }
};

function App() {
  const [currentStep, setCurrentStep] = useState('search'); // search, select, initialize, chat
  const [language, setLanguage] = useState('en'); // en, pl
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: TRANSLATIONS['en'].welcomeMessage,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedGame, setSelectedGame] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [sessionStatus, setSessionStatus] = useState(null);
  const messagesEndRef = useRef(null);
  const loadingIntervalRef = useRef(null);
  const statusCheckIntervalRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isLoading) {
      let messageIndex = 0;
      setLoadingMessage(FUNNY_LOADING_MESSAGES[messageIndex]);

      loadingIntervalRef.current = setInterval(() => {
        messageIndex = (messageIndex + 1) % FUNNY_LOADING_MESSAGES.length;
        setLoadingMessage(FUNNY_LOADING_MESSAGES[messageIndex]);
      }, 2500);
    } else {
      if (loadingIntervalRef.current) {
        clearInterval(loadingIntervalRef.current);
        loadingIntervalRef.current = null;
      }
      setLoadingMessage('');
    }

    return () => {
      if (loadingIntervalRef.current) {
        clearInterval(loadingIntervalRef.current);
      }
    };
  }, [isLoading]);

  const searchGames = async (query) => {
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      const response = await axios.post('/api/v1/games/search', {
        query: query.trim(),
        max_results: 5
      });

      if (response.data.success && response.data.games.length > 0) {
        setSearchResults(response.data.games);
        setCurrentStep('select');

        const resultsMessage = {
          id: Date.now(),
          type: 'bot',
          content: t('foundGames', { count: response.data.games.length, query: query }),
          timestamp: new Date()
        };
        setMessages(prev => [...prev, resultsMessage]);
      } else {
        const noResultsMessage = {
          id: Date.now(),
          type: 'bot',
          content: t('noGamesFound', { query: query }),
          timestamp: new Date()
        };
        setMessages(prev => [...prev, noResultsMessage]);
      }
    } catch (error) {
      console.error('Error searching games:', error);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        content: t('searchError'),
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const selectGame = async (game) => {
    setSelectedGame(game);
    setCurrentStep('initialize');

    const selectionMessage = {
      id: Date.now(),
      type: 'bot',
      content: t('gameSelected', { name: game.name }),
      timestamp: new Date()
    };
    setMessages(prev => [...prev, selectionMessage]);

    // Initialize session
    setIsLoading(true);
    try {
      console.log('Initializing session for game:', game);
      const response = await axios.post('/api/v1/sessions/init', {
        appid: game.appid
      });

      console.log('Session init response:', response.data);

      if (response.data && response.data.success && response.data.session_info) {
        setSessionId(response.data.session_info.session_id);
        setSessionStatus(response.data.session_info.status);

        // If session is already ready, don't start polling
        if (response.data.session_info.status === 'ready') {
          setIsLoading(false);
          setCurrentStep('chat');

          const readyMessage = {
            id: Date.now(),
            type: 'bot',
            content: t('existingSession', { count: response.data.session_info.reviews_count, name: game.name }),
            timestamp: new Date()
          };
          setMessages(prev => [...prev, readyMessage]);
        } else {
          // Start polling for status updates
          startStatusPolling(response.data.session_info.session_id);
        }
      } else {
        throw new Error(response.data?.message || 'Invalid response from server');
      }
    } catch (error) {
      console.error('Error initializing session:', error, error.response?.data);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        content: t('initError', { name: game.name, error: error.response?.data?.error_message || error.message || 'Unknown error' }),
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  const startStatusPolling = (sessionId) => {
    let pollCount = 0;
    const maxPolls = 200; // Maximum 10 minutes of polling (200 * 3 seconds)

    statusCheckIntervalRef.current = setInterval(async () => {
      pollCount++;

      try {
        console.log(`Polling session status (attempt ${pollCount}):`, sessionId);
        const response = await axios.get(`/api/v1/sessions/${sessionId}/status`);

        console.log('Status response:', response.data);

        if (response.data && response.data.success && response.data.session_info) {
          const status = response.data.session_info.status;
          const progress = response.data.session_info.progress_percentage || 0;
          const reviewsCount = response.data.session_info.reviews_count || 0;

          setSessionStatus(status);

          // Update loading message based on status
          const progressText = progress > 0 ? ` (${Math.round(progress)}%)` : '';
          const reviewText = reviewsCount > 0 ? ` - ${reviewsCount} ${language === 'pl' ? 'recenzji' : 'reviews'}` : '';
          // The loading message rotation will continue with FUNNY_LOADING_MESSAGES and show progress
          if (progress > 0 || reviewsCount > 0) {
            setLoadingMessage(`${FUNNY_LOADING_MESSAGES[Math.floor(Math.random() * FUNNY_LOADING_MESSAGES.length)]}${progressText}${reviewText}`);
          }

          if (status === 'ready') {
            console.log('Session is ready!');
            clearInterval(statusCheckIntervalRef.current);
            setIsLoading(false);
            setCurrentStep('chat');

            const readyMessage = {
              id: Date.now(),
              type: 'bot',
              content: t('sessionReady', { count: reviewsCount, name: selectedGame.name }),
              timestamp: new Date()
            };
            setMessages(prev => [...prev, readyMessage]);
          } else if (status === 'error') {
            console.log('Session error:', response.data.session_info.error_message);
            clearInterval(statusCheckIntervalRef.current);
            setIsLoading(false);

            const errorMessage = {
              id: Date.now(),
              type: 'bot',
              content: t('sessionError', { name: selectedGame.name, error: response.data.session_info.error_message || 'Please try again later.' }),
              timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMessage]);
          }
        } else {
          console.warn('Unexpected response format:', response.data);
          // Don't stop polling for unexpected formats, might be temporary
          if (pollCount >= maxPolls) {
            clearInterval(statusCheckIntervalRef.current);
            setIsLoading(false);
            const timeoutMessage = {
              id: Date.now(),
              type: 'bot',
              content: t('processingTimeout'),
              timestamp: new Date()
            };
            setMessages(prev => [...prev, timeoutMessage]);
          }
        }
      } catch (error) {
        console.error('Error checking session status:', error, error.response?.data);

        // Check for specific error types
        if (error.response?.status === 404) {
          clearInterval(statusCheckIntervalRef.current);
          setIsLoading(false);
          const notFoundMessage = {
            id: Date.now(),
            type: 'bot',
            content: t('sessionNotFound'),
            timestamp: new Date()
          };
          setMessages(prev => [...prev, notFoundMessage]);
        } else if (error.response?.status >= 500) {
          // Server error - don't stop polling immediately, might be temporary
          console.log('Server error, continuing to poll...');
          if (pollCount >= maxPolls) {
            clearInterval(statusCheckIntervalRef.current);
            setIsLoading(false);
            const serverErrorMessage = {
              id: Date.now(),
              type: 'bot',
              content: t('serverError'),
              timestamp: new Date()
            };
            setMessages(prev => [...prev, serverErrorMessage]);
          }
        } else if (pollCount >= maxPolls) {
          // Too many failed attempts
          clearInterval(statusCheckIntervalRef.current);
          setIsLoading(false);
          const connectionMessage = {
            id: Date.now(),
            type: 'bot',
            content: t('connectionLost'),
            timestamp: new Date()
          };
          setMessages(prev => [...prev, connectionMessage]);
        }
        // For other errors, continue polling (might be temporary network issues)
      }
    }, 3000); // Check every 3 seconds
  };

  const askQuestion = async (question) => {
    if (!question.trim() || !sessionId || currentStep !== 'chat') {
      console.warn('Cannot ask question:', { question: question.trim(), sessionId, currentStep });
      return;
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: question,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      console.log('Asking question:', { sessionId, question: question.trim(), language });
      const response = await axios.post(`/api/v1/sessions/${sessionId}/question`, {
        question: question.trim(),
        language: language
      });

      console.log('Question response:', response.data);

      if (response.data && response.data.success && response.data.answer) {
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: response.data.answer,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error(response.data?.message || 'No answer received from server');
      }
    } catch (error) {
      console.error('Error asking question:', error, error.response?.data);

      let errorContent = t('questionError');

      if (error.response?.status === 404) {
        errorContent = t('sessionNotFound');
      } else if (error.response?.status === 429) {
        errorContent = t('tooManyRequests');
      } else if (error.response?.status >= 500) {
        errorContent = t('serverError');
      } else if (error.response?.data?.error_message) {
        errorContent = `${t('questionError')} ${error.response.data.error_message}`;
      } else if (error.message) {
        errorContent = `${t('questionError')} ${error.message}`;
      }

      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: errorContent,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => {
    if (!inputValue.trim() || isLoading) return;

    if (currentStep === 'search') {
      const userMessage = {
        id: Date.now(),
        type: 'user',
        content: inputValue,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, userMessage]);
      searchGames(inputValue);
      setInputValue('');
    } else if (currentStep === 'chat') {
      askQuestion(inputValue);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const startOver = () => {
    setCurrentStep('search');
    setSearchResults([]);
    setSelectedGame(null);
    setSessionId(null);
    setSessionStatus(null);
    setIsLoading(false);

    if (statusCheckIntervalRef.current) {
      clearInterval(statusCheckIntervalRef.current);
    }

    const restartMessage = {
      id: Date.now(),
      type: 'bot',
      content: t('restartMessage'),
      timestamp: new Date()
    };
    setMessages(prev => [...prev, restartMessage]);
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getPlaceholderText = () => {
    if (currentStep === 'search') {
      return TRANSLATIONS[language].searchPlaceholder;
    } else if (currentStep === 'chat') {
      return TRANSLATIONS[language].chatPlaceholder;
    }
    return "";
  };

  const t = (key, params = {}) => {
    let text = TRANSLATIONS[language][key] || key;
    Object.keys(params).forEach(param => {
      text = text.replace(`{${param}}`, params[param]);
    });
    return text;
  };

  const changeLanguage = (newLanguage) => {
    setLanguage(newLanguage);

    // Update welcome message
    const welcomeMessage = {
      id: 1,
      type: 'bot',
      content: TRANSLATIONS[newLanguage].welcomeMessage,
      timestamp: new Date()
    };

    setMessages([welcomeMessage]);

    // Reset to search step
    setCurrentStep('search');
    setSearchResults([]);
    setSelectedGame(null);
    setSessionId(null);
    setSessionStatus(null);
    setIsLoading(false);

    if (statusCheckIntervalRef.current) {
      clearInterval(statusCheckIntervalRef.current);
    }
  };

  const isInputDisabled = () => {
    return isLoading || currentStep === 'select' || currentStep === 'initialize';
  };

  // Cleanup intervals on unmount
  useEffect(() => {
    return () => {
      if (statusCheckIntervalRef.current) {
        clearInterval(statusCheckIntervalRef.current);
      }
    };
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="header-main">
            <h1>🔍 {t('title')}</h1>
            <p>{t('subtitle')}</p>
            <div className="header-actions">
              {currentStep !== 'search' && (
                <button className="start-over-btn" onClick={startOver}>
                  🔄 {t('startOver')}
                </button>
              )}
              <div className="language-selector">
                <button
                  className={`lang-btn ${language === 'en' ? 'active' : ''}`}
                  onClick={() => changeLanguage('en')}
                >
                  EN
                </button>
                <button
                  className={`lang-btn ${language === 'pl' ? 'active' : ''}`}
                  onClick={() => changeLanguage('pl')}
                >
                  PL
                </button>
              </div>
              <a
                href="https://github.com/VerusLucian/SteamLensAI"
                target="_blank"
                rel="noopener noreferrer"
                className="github-link"
                title="View on GitHub"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/>
                </svg>
              </a>
            </div>
          </div>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages-container">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.type}-message`}>
              <div className="message-avatar">
                {message.type === 'bot' ? '🤖' : '👤'}
              </div>
              <div className="message-content">
                <div className="message-text">
                  <ReactMarkdown
                    components={markdownComponents}
                    disallowedElements={['script', 'iframe', 'object', 'embed']}
                    unwrapDisallowed={true}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
                <div className="message-time">{formatTime(message.timestamp)}</div>
              </div>
            </div>
          ))}

          {currentStep === 'select' && searchResults.length > 0 && (
            <div className="game-selection">
              {searchResults.map((game) => (
                <div key={game.appid} className="game-card" onClick={() => selectGame(game)}>
                  <div className="game-header">
                    {game.header_image && (
                      <img src={game.header_image} alt={game.name} className="game-image" />
                    )}
                  </div>
                  <div className="game-info">
                    <h3>{game.name}</h3>
                    <p className="game-details">
                      {game.developers.length > 0 && `By ${game.developers.join(', ')}`}
                      {game.release_date && ` • Released ${game.release_date}`}
                    </p>
                    {game.short_description && (
                      <p className="game-description">{game.short_description}</p>
                    )}
                    <div className="game-meta">
                      {game.price && <span className="game-price">{game.price}</span>}
                      {game.genres.length > 0 && (
                        <span className="game-genres">{game.genres.slice(0, 3).join(', ')}</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {isLoading && (
            <div className="message bot-message loading-message">
              <div className="message-avatar">
                <div className="loading-spinner">🤖</div>
              </div>
              <div className="message-content">
                <div className="message-text">
                  <ReactMarkdown
                    components={markdownComponents}
                    disallowedElements={['script', 'iframe', 'object', 'embed']}
                    unwrapDisallowed={true}
                  >
                    {loadingMessage}
                  </ReactMarkdown>
                </div>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {(currentStep === 'search' || currentStep === 'chat') && (
          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={getPlaceholderText()}
                rows="1"
                disabled={isInputDisabled()}
                className="message-input"
              />
              <button
                onClick={handleSubmit}
                disabled={!inputValue.trim() || isInputDisabled()}
                className="send-button"
              >
                {isLoading ? '⏳' : currentStep === 'search' ? '🔍' : '🚀'}
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
