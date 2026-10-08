/* ==========================================================================
   script.js — Andrew Ng Digital Twin Frontend Logic
   Uses fetch() to communicate with the FastAPI backend (/api/chat, /api/reset, etc.)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatContainer = document.getElementById('chat-container');
    const messagesList = document.getElementById('messages-list');
    const welcomeCard = document.getElementById('welcome-card');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const sendIcon = sendBtn.querySelector('.send-icon');
    const sendSpinner = document.getElementById('send-spinner');
    const micBtn = document.getElementById('mic-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const ttsToggleBtn = document.getElementById('tts-toggle-btn');
    const historyModalBtn = document.getElementById('history-modal-btn');
    const memoryModal = document.getElementById('memory-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalBodyContent = document.getElementById('modal-body-content');
    const toastContainer = document.getElementById('toast-container');
    const statusText = document.getElementById('status-text');

    // App State
    let conversationHistory = [];
    let isProcessing = false;
    let autoTTS = false;
    let recognition = null;

    // Determine API Base URL dynamically
    const API_BASE = window.location.origin.includes('http') ? window.location.origin : 'http://localhost:8000';

    // Initial Health Check
    checkSystemStatus();

    // --------------------------------------------------------------------------
    // Event Listeners
    // --------------------------------------------------------------------------
    
    // Auto-expand textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });

    // Enter to submit (Shift+Enter for newline)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    sendBtn.addEventListener('click', handleSendMessage);

    // Prompt Chips Click Handlers
    document.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const promptText = chip.getAttribute('data-prompt');
            if (promptText && !isProcessing) {
                userInput.value = promptText;
                handleSendMessage();
            }
        });
    });

    // Reset / Clear Chat
    clearChatBtn.addEventListener('click', async () => {
        if (isProcessing) return;
        try {
            const res = await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
            if (res.ok) {
                conversationHistory = [];
                messagesList.innerHTML = '';
                welcomeCard.classList.remove('hidden');
                showToast('Chat session and short-term memory reset!');
            }
        } catch (err) {
            showToast('Error resetting memory', 'error');
        }
    });

    // Auto Text-To-Speech Toggle
    ttsToggleBtn.addEventListener('click', () => {
        autoTTS = !autoTTS;
        ttsToggleBtn.classList.toggle('active', autoTTS);
        showToast(autoTTS ? 'Auto Text-to-Speech enabled' : 'Auto Text-to-Speech disabled');
    });

    // Memory Logs Modal
    historyModalBtn.addEventListener('click', openMemoryModal);
    modalCloseBtn.addEventListener('click', () => memoryModal.classList.add('hidden'));
    memoryModal.addEventListener('click', (e) => {
        if (e.target === memoryModal) memoryModal.classList.add('hidden');
    });

    // Web Speech Recognition (Mic Button)
    setupSpeechRecognition();

    // --------------------------------------------------------------------------
    // Main Chat Functions
    // --------------------------------------------------------------------------

    async function handleSendMessage() {
        const text = userInput.value.trim();
        if (!text || isProcessing) return;

        // Hide welcome card on first message
        if (!welcomeCard.classList.contains('hidden')) {
            welcomeCard.classList.add('hidden');
        }

        // Render User Message
        appendMessage('user', text);
        userInput.value = '';
        userInput.style.height = 'auto';

        // Show Typing Indicator
        const typingElement = appendTypingIndicator();
        setLoadingState(true);

        try {
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: text,
                    history: conversationHistory,
                    include_sources: true
                })
            });

            // Remove typing indicator
            typingElement.remove();

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const errorMsg = errorData.detail || `Server error (${response.status})`;
                appendMessage('assistant', `⚠️ ${errorMsg}`, [], 0);
                showToast(errorMsg, 'error');
                return;
            }

            const data = await response.json();

            // Update conversation history state
            conversationHistory.push({
                user: text,
                model: data.response
            });

            // Render Assistant Message with RAG Sources & Timing
            appendMessage('assistant', data.response, data.sources, data.elapsed_seconds);

            // Auto Speak if enabled
            if (autoTTS && data.response) {
                speakText(data.response);
            }

        } catch (error) {
            if (typingElement) typingElement.remove();
            console.error('Chat API Error:', error);
            appendMessage('assistant', '⚠️ Failed to connect to Python backend server. Make sure `python api.py` is running.');
            showToast('Connection error to Python API', 'error');
        } finally {
            setLoadingState(false);
            scrollToBottom();
        }
    }

    // --------------------------------------------------------------------------
    // UI Render Helpers
    // --------------------------------------------------------------------------

    function appendMessage(role, text, sources = [], elapsedSeconds = 0) {
        const messageRow = document.createElement('div');
        messageRow.className = `message-row ${role}`;

        if (role === 'assistant') {
            const avatar = document.createElement('div');
            avatar.className = 'msg-avatar';
            avatar.innerHTML = '🎓';
            messageRow.appendChild(avatar);
        }

        const container = document.createElement('div');
        container.className = 'msg-bubble-container';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.textContent = text;
        container.appendChild(bubble);

        // Metadata & Actions for Assistant
        if (role === 'assistant') {
            const meta = document.createElement('div');
            meta.className = 'msg-meta';

            if (elapsedSeconds > 0) {
                const timeSpan = document.createElement('span');
                timeSpan.className = 'time-badge';
                timeSpan.textContent = `⚡ ${elapsedSeconds}s`;
                meta.appendChild(timeSpan);
            }

            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'msg-actions';

            // Copy button
            const copyBtn = document.createElement('button');
            copyBtn.className = 'action-icon-btn';
            copyBtn.innerHTML = '📋 Copy';
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(text);
                showToast('Copied response to clipboard');
            };
            actionsDiv.appendChild(copyBtn);

            // Speak button
            const speakBtn = document.createElement('button');
            speakBtn.className = 'action-icon-btn';
            speakBtn.innerHTML = '🔊 Speak';
            speakBtn.onclick = () => speakText(text);
            actionsDiv.appendChild(speakBtn);

            meta.appendChild(actionsDiv);
            container.appendChild(meta);

            // RAG Sources Accordion
            if (sources && sources.length > 0) {
                const sourcesContainer = document.createElement('div');
                sourcesContainer.className = 'sources-container';

                const sourcesHeader = document.createElement('div');
                sourcesHeader.className = 'sources-header';
                sourcesHeader.innerHTML = `<span>🔍 Retrieved RAG Context (${sources.length} sources)</span> <span>▼</span>`;

                const sourcesContent = document.createElement('div');
                sourcesContent.className = 'sources-list-content hidden';

                sources.forEach((src, idx) => {
                    const srcItem = document.createElement('div');
                    srcItem.className = 'source-item';
                    srcItem.innerHTML = `
                        <div class="source-item-header">
                            <span>Chunk #${idx + 1} (${src.id})</span>
                            <span class="source-score">Rerank: ${src.rerank_score}</span>
                        </div>
                        <div>"${escapeHtml(src.document)}"</div>
                    `;
                    sourcesContent.appendChild(srcItem);
                });

                sourcesHeader.onclick = () => {
                    const isHidden = sourcesContent.classList.toggle('hidden');
                    sourcesHeader.querySelector('span:last-child').textContent = isHidden ? '▼' : '▲';
                };

                sourcesContainer.appendChild(sourcesHeader);
                sourcesContainer.appendChild(sourcesContent);
                container.appendChild(sourcesContainer);
            }
        }

        messageRow.appendChild(container);
        messagesList.appendChild(messageRow);
        scrollToBottom();
        return messageRow;
    }

    function appendTypingIndicator() {
        const messageRow = document.createElement('div');
        messageRow.className = 'message-row assistant';

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.innerHTML = '🎓';
        messageRow.appendChild(avatar);

        const container = document.createElement('div');
        container.className = 'msg-bubble-container';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble typing-indicator';
        bubble.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;

        container.appendChild(bubble);
        messageRow.appendChild(container);
        messagesList.appendChild(messageRow);
        scrollToBottom();
        return messageRow;
    }

    function setLoadingState(loading) {
        isProcessing = loading;
        sendBtn.disabled = loading;
        if (loading) {
            sendIcon.classList.add('hidden');
            sendSpinner.classList.remove('hidden');
        } else {
            sendIcon.classList.remove('hidden');
            sendSpinner.classList.add('hidden');
        }
    }

    function scrollToBottom() {
        const viewport = document.querySelector('.chat-viewport');
        viewport.scrollTop = viewport.scrollHeight;
    }

    function escapeHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // --------------------------------------------------------------------------
    // Toast & Modal Helpers
    // --------------------------------------------------------------------------

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    async function checkSystemStatus() {
        try {
            const res = await fetch(`${API_BASE}/api/status`);
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'ok') {
                    statusText.textContent = `Online (${data.indexed_documents_count} Chunks)`;
                } else if (data.status === 'warning_missing_api_key') {
                    statusText.textContent = `⚠️ Missing GEMINI_API_KEY`;
                    statusText.parentElement.style.color = '#F59E0B';
                }
            }
        } catch (e) {
            statusText.textContent = 'API Offline';
            statusText.parentElement.style.color = '#EF4444';
        }
    }

    async function openMemoryModal() {
        memoryModal.classList.remove('hidden');
        modalBodyContent.innerHTML = '<div class="loading-state">Fetching memory state...</div>';

        try {
            const res = await fetch(`${API_BASE}/api/history`);
            if (res.ok) {
                const data = await res.json();
                modalBodyContent.innerHTML = `
                    <p style="margin-bottom: 8px; font-weight: 600; color: #3B82F6;">Short-Term Memory (${data.short_term_memory.length} turns):</p>
                    <div class="json-block">${escapeHtml(JSON.stringify(data.short_term_memory, null, 2))}</div>
                    
                    <p style="margin-top: 16px; margin-bottom: 8px; font-weight: 600; color: #10B981;">Long-Term Memory Summaries:</p>
                    <div class="json-block">${escapeHtml(JSON.stringify(data.long_term_memory, null, 2))}</div>
                `;
            } else {
                modalBodyContent.innerHTML = '<div class="error">Failed to load memory state.</div>';
            }
        } catch (err) {
            modalBodyContent.innerHTML = '<div class="error">Error connecting to server.</div>';
        }
    }

    // Speech Synthesis
    function speakText(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel(); // Stop any active speech
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        } else {
            showToast('Speech synthesis not supported in this browser.', 'error');
        }
    }

    // Voice Input via Web Speech API
    function setupSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            micBtn.title = 'Speech recognition not supported in browser';
            micBtn.style.opacity = '0.5';
            return;
        }

        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        micBtn.addEventListener('click', () => {
            if (micBtn.classList.contains('listening')) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });

        recognition.onstart = () => {
            micBtn.classList.add('listening');
            showToast('Listening... Speak now.');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            showToast(`Recognized: "${transcript}"`);
            handleSendMessage();
        };

        recognition.onerror = (event) => {
            console.error('Speech error:', event.error);
            showToast(`Speech recognition error: ${event.error}`, 'error');
            micBtn.classList.remove('listening');
        };

        recognition.onend = () => {
            micBtn.classList.remove('listening');
        };
    }
});
