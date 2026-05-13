/**
 * Transformer Chatbot – Toán Rời Rạc
 * Frontend Application Logic
 * v2.0 – Markdown + KaTeX + Conversation Memory
 */

const API_BASE = '';  // Same origin

// ─── DOM Elements ───
const chatContainer = document.getElementById('chatContainer');
const chatMessages = document.getElementById('chatMessages');
const welcomeScreen = document.getElementById('welcomeScreen');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');
const newChatBtn = document.getElementById('newChatBtn');
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const clearBtn = document.getElementById('clearBtn');

// ─── State ───
let isWaiting = false;
let conversationMessages = [];    // UI display list (kept for backward compat)
let conversationHistory = [];     // Memory: [{role:'user'|'bot', content:'...'}]

// ─── Marked.js config ───
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,     // GFM line breaks
        gfm: true,        // GitHub Flavored Markdown
    });
}

// ─── Init ───
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    autoResizeInput();
    chatInput.focus();
});

// ─── Health Check ───
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        const badge = document.getElementById('modelBadge');
        if (data.model_loaded) {
            badge.innerHTML = `<span class="pulse"></span> Model sẵn sàng · ${data.model_info.device.toUpperCase()}`;
        } else {
            badge.innerHTML = `<span class="pulse" style="background:var(--accent-danger)"></span> Model chưa load`;
            badge.style.borderColor = 'rgba(244,63,94,0.15)';
            badge.style.color = 'var(--accent-danger)';
        }
    } catch {
        const badge = document.getElementById('modelBadge');
        badge.innerHTML = `<span class="pulse" style="background:var(--accent-danger)"></span> Server offline`;
        badge.style.borderColor = 'rgba(244,63,94,0.15)';
        badge.style.color = 'var(--accent-danger)';
    }
}

// ─── Auto Resize Textarea ───
function autoResizeInput() {
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
    });
}

// ─── Send Message ───
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isWaiting) return;

    // Hide welcome, show chat
    if (welcomeScreen) welcomeScreen.style.display = 'none';

    // Add user message to UI & history
    addMessage('user', message);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.focus();

    // Show typing
    isWaiting = true;
    sendBtn.disabled = true;
    typingIndicator.classList.add('active');
    scrollToBottom();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: conversationHistory,
                temperature: 0.5,
                top_k: 3,
                max_tokens: 100,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        typingIndicator.classList.remove('active');

        addMessage('bot', data.response, {
            tokens: data.tokens_generated,
            time: data.inference_time_ms,
            source_layer: data.source_layer,
            source_method: data.source_method,
            confidence: data.confidence,
            matched_question: data.matched_question,
        });
    } catch (err) {
        typingIndicator.classList.remove('active');
        addMessage('bot', `⚠️ Không thể kết nối: ${err.message}`, null, true);
        showError(err.message);
    } finally {
        isWaiting = false;
        sendBtn.disabled = false;
    }
}

// ─── Add Message to Chat ───
function addMessage(role, content, meta = null, isError = false) {
    const now = new Date();
    const timeStr = now.getHours().toString().padStart(2, '0') + ':' +
                    now.getMinutes().toString().padStart(2, '0');

    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🤖';
    const sender = role === 'user' ? 'Bạn' : 'Transformer Bot';

    let metaHTML = '';
    if (meta && role === 'bot') {
        // Tạo badge hiển thị Layer nguồn gốc
        let layerBadge = '';
        if (meta.source_layer === 1) {
            layerBadge = `<span class="meta-badge layer-badge layer-1">🛡️ Lớp 1 · Chính xác</span>`;
        } else if (meta.source_layer === 2) {
            layerBadge = `<span class="meta-badge layer-badge layer-2">🔮 Lớp 2 · Ngữ nghĩa</span>`;
        } else if (meta.source_layer === 3) {
            layerBadge = `<span class="meta-badge layer-badge layer-3">🧬 Lớp 3 · AI Sáng tạo</span>`;
        }

        // Badge độ tin cậy (chỉ hiện cho Layer 1 & 2)
        let confidenceBadge = '';
        if (meta.confidence != null) {
            const pct = Math.round(meta.confidence * 100);
            confidenceBadge = `<span class="meta-badge">🎯 ${pct}%</span>`;
        }

        metaHTML = `
            <div class="message-meta">
                ${layerBadge}
                ${confidenceBadge}
                <span class="meta-badge">⚡ ${meta.time}ms</span>
                <span class="meta-badge">📝 ${meta.tokens} tokens</span>
                <button class="copy-btn" onclick="copyText(this)" title="Sao chép">📋</button>
            </div>
        `;
    }

    const contentClass = isError ? 'style="color: var(--accent-danger);"' : '';

    msgEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-body">
            <div class="message-sender">
                ${sender}
                <span class="message-time">${timeStr}</span>
            </div>
            <div class="message-content" ${contentClass}>${formatContent(content)}</div>
            ${metaHTML}
        </div>
    `;

    // Store raw content as data attribute for copy
    msgEl.dataset.rawContent = content;

    chatMessages.appendChild(msgEl);

    // Render KaTeX math in the newly added message
    try {
        if (typeof renderMathInElement === 'function') {
            const contentEl = msgEl.querySelector('.message-content');
            renderMathInElement(contentEl, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                ],
                throwOnError: false,
            });
        }
    } catch (_) { /* KaTeX not loaded yet, skip silently */ }

    // Push to both arrays
    conversationMessages.push({ role, content });
    if (!isError) {
        conversationHistory.push({ role, content });
    }

    scrollToBottom();
}

// ─── Format Content (Markdown via marked.js) ───
function formatContent(text) {
    if (typeof marked !== 'undefined' && marked.parse) {
        // Use marked.js for full Markdown rendering
        const rawHtml = marked.parse(text);
        // Wrap in a container to avoid paragraph margin issues
        return rawHtml;
    }
    // Fallback: basic formatting if marked.js not loaded
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`(.*?)`/g, '<code style="background:rgba(124,90,255,0.12);padding:1px 5px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:12px;color:var(--accent-tertiary);">$1</code>');
    html = html.replace(/\n/g, '<br>');
    return html;
}

// ─── Escape HTML ───
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── Copy Text ───
function copyText(btn) {
    // Walk up to the .message element and grab the raw content
    const msgEl = btn.closest('.message');
    const raw = msgEl ? msgEl.dataset.rawContent : '';

    navigator.clipboard.writeText(raw).then(() => {
        const original = btn.textContent;
        btn.textContent = '✅';
        setTimeout(() => btn.textContent = original, 1500);
    });
}

// ─── Scroll ───
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}

// ─── Suggestion Click ───
function askSuggestion(text) {
    chatInput.value = text;
    sendMessage();
}

// ─── New Chat ───
function newChat() {
    chatMessages.innerHTML = '';
    conversationMessages = [];
    conversationHistory = [];
    if (welcomeScreen) welcomeScreen.style.display = 'flex';
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.focus();
    closeSidebar();
}

// ─── Error Toast ───
function showError(msg) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = `❌ ${msg}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ─── Sidebar Mobile ───
function toggleSidebar() {
    sidebar.classList.toggle('open');
    sidebarOverlay.classList.toggle('active');
}

function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('active');
}

// ─── Keyboard Events ───
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);
newChatBtn.addEventListener('click', newChat);
menuToggle.addEventListener('click', toggleSidebar);
sidebarOverlay.addEventListener('click', closeSidebar);
clearBtn.addEventListener('click', newChat);

// Topic click
document.querySelectorAll('.topic-item').forEach(item => {
    item.addEventListener('click', () => {
        const q = item.dataset.question;
        if (q) {
            chatInput.value = q;
            closeSidebar();
            sendMessage();
        }
    });
});
