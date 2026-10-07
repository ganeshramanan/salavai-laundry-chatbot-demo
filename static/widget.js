/**
 * Multi-Tenant Live Chat & Bot Widget
 * 
 * Embed usage on any website (Salavai Laundry, or any of Hari's 4 sites):
 * <script src="https://salavai-laundry-chatbot-demo-1.onrender.com/widget.js" data-site="salavai"></script>
 */
(function () {
  // 1. Detect configuration from current script tag
  const currentScript = document.currentScript || (function () {
    const scripts = document.getElementsByTagName("script");
    return scripts[scripts.length - 1];
  })();

  const SITE_ID = (currentScript && currentScript.getAttribute("data-site")) || "salavai";

  const API_BASE = window.location.origin.includes("localhost")
    ? "http://localhost:5008"
    : "https://salavai-laundry-chatbot-demo-1.onrender.com";

  let sessionId = localStorage.getItem("slw_session_" + SITE_ID) || null;
  let pollInterval = null;
  let isLiveMode = false;

  // Default theme fallback
  let theme = {
    primary: "#a41e22",
    dark: "#14324f",
    name: "Customer Support",
    greeting: "👋 Hello! How can we assist you today?",
    quickReplies: ["Pricing", "💬 Talk to Human"]
  };

  // 2. Inject CSS
  const style = document.createElement("style");
  style.id = "slw-custom-styles";
  style.textContent = `
    #slw-bubble {
      position: fixed; bottom: 24px; right: 24px; width: 62px; height: 62px; border-radius: 50%;
      background: #a41e22; color: #fff; display: flex; align-items: center; justify-content: center;
      font-size: 28px; cursor: pointer; box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 999999;
      transition: transform .2s, box-shadow .2s; border: none; font-family: -apple-system, sans-serif;
    }
    #slw-bubble:hover { transform: scale(1.08); box-shadow: 0 6px 24px rgba(0,0,0,0.4); }
    #slw-window {
      position: fixed; bottom: 98px; right: 24px; width: 370px; height: 540px; max-height: 82vh;
      background: #fff; border-radius: 16px; box-shadow: 0 8px 40px rgba(0,0,0,0.25);
      display: none; flex-direction: column; overflow: hidden; z-index: 999999;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #slw-window.slw-open { display: flex; }
    #slw-header {
      background: #14324f; color: #fff; padding: 16px; font-weight: 700; font-size: 14px;
      display: flex; justify-content: space-between; align-items: center;
    }
    #slw-header .slw-sub { font-size: 11px; color: #a8b8c8; font-weight: 400; margin-top: 2px; }
    #slw-status-badge {
      display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 10px;
      background: #059669; color: #fff; margin-left: 6px; font-weight: 600;
    }
    #slw-close { cursor: pointer; font-size: 18px; opacity: 0.8; }
    #slw-close:hover { opacity: 1; }
    #slw-messages {
      flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px;
      background: #f9fafb;
    }
    .slw-msg { max-width: 85%; padding: 10px 14px; border-radius: 14px; font-size: 13px; line-height: 1.5; white-space: pre-line; word-break: break-word; }
    .slw-msg-bot { background: #fff; border: 1px solid #e5e7eb; align-self: flex-start; color: #1f2937; }
    .slw-msg-agent { background: #eff6ff; border: 1px solid #bfdbfe; align-self: flex-start; color: #1e3a8a; }
    .slw-msg-user { background: #a41e22; color: #fff; align-self: flex-end; }
    .slw-msg-agent-tag { font-size: 10px; font-weight: 700; color: #2563eb; margin-bottom: 2px; }
    #slw-quick { display: flex; gap: 6px; flex-wrap: wrap; padding: 0 16px 10px; background: #f9fafb; }
    .slw-quick-reply {
      font-size: 11px; border: 1px solid #a41e22; color: #a41e22; background: #fff;
      padding: 6px 12px; border-radius: 14px; cursor: pointer; font-weight: 500;
    }
    .slw-quick-reply:hover { background: #a41e22; color: #fff; }
    #slw-input-row { display: flex; gap: 8px; padding: 12px; border-top: 1px solid #e5e7eb; background: #fff; }
    #slw-input { flex: 1; border: 1px solid #e5e7eb; border-radius: 20px; padding: 10px 14px; font-size: 13px; }
    #slw-input:focus { outline: none; border-color: #a41e22; }
    #slw-send {
      background: #a41e22; color: #fff; border: none; border-radius: 50%; width: 40px; height: 40px;
      cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center;
    }
    .slw-live-banner {
      background: #ecfdf5; border-bottom: 1px solid #a7f3d0; color: #065f46;
      font-size: 11px; padding: 6px 16px; text-align: center; font-weight: 500;
    }
  `;
  document.head.appendChild(style);

  // 3. Inject DOM elements
  const bubble = document.createElement("button");
  bubble.id = "slw-bubble";
  bubble.textContent = "💬";
  document.body.appendChild(bubble);

  const chatWindow = document.createElement("div");
  chatWindow.id = "slw-window";
  chatWindow.innerHTML = `
    <div id="slw-header">
      <div>
        <span id="slw-header-title">Support Assistant</span>
        <span id="slw-status-badge">Online</span>
        <div class="slw-sub" id="slw-header-sub">Ask questions or chat with live agent</div>
      </div>
      <div id="slw-close">✕</div>
    </div>
    <div id="slw-banner" style="display:none;" class="slw-live-banner">🟢 Live Agent Connected</div>
    <div id="slw-messages"></div>
    <div id="slw-quick"></div>
    <div id="slw-input-row">
      <input type="text" id="slw-input" placeholder="Type your message...">
      <button id="slw-send">➤</button>
    </div>
  `;
  document.body.appendChild(chatWindow);

  // 4. Fetch site configuration & customize colors
  fetch(API_BASE + "/api/site-config?site_id=" + encodeURIComponent(SITE_ID))
    .then(r => r.json())
    .then(cfg => {
      theme = cfg;
      applyTheme(cfg);
    })
    .catch(() => {
      applyTheme(theme);
    });

  function applyTheme(cfg) {
    document.getElementById("slw-header-title").textContent = cfg.name;
    document.getElementById("slw-bubble").style.background = cfg.primary_color;
    document.getElementById("slw-header").style.background = cfg.dark_color;
    document.getElementById("slw-send").style.background = cfg.primary_color;

    // Greeting message
    const msgContainer = document.getElementById("slw-messages");
    if (!msgContainer.hasChildNodes()) {
      appendMessage(cfg.greeting, "bot");
    }

    // Quick replies
    const quickContainer = document.getElementById("slw-quick");
    quickContainer.innerHTML = "";
    (cfg.quick_replies || []).forEach(text => {
      const btn = document.createElement("div");
      btn.className = "slw-quick-reply";
      btn.style.borderColor = cfg.primary_color;
      btn.style.color = cfg.primary_color;
      btn.textContent = text;
      btn.addEventListener("click", () => sendMessage(text));
      quickContainer.appendChild(btn);
    });
  }

  function appendMessage(text, sender) {
    const container = document.getElementById("slw-messages");
    const div = document.createElement("div");

    if (sender === "user") {
      div.className = "slw-msg slw-msg-user";
      div.style.background = theme.primary_color || "#a41e22";
      div.textContent = text;
    } else if (sender === "agent") {
      div.className = "slw-msg slw-msg-agent";
      div.innerHTML = `<div class="slw-msg-agent-tag">Support Agent (Live)</div>${escapeHtml(text)}`;
    } else {
      div.className = "slw-msg slw-msg-bot";
      div.textContent = text;
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  function escapeHtml(str) {
    const p = document.createElement("p");
    p.textContent = str;
    return p.innerHTML;
  }

  function setLiveModeUI(live) {
    isLiveMode = live;
    const banner = document.getElementById("slw-banner");
    const badge = document.getElementById("slw-status-badge");
    if (live) {
      banner.style.display = "block";
      badge.textContent = "Live Agent";
      badge.style.background = "#2563eb";
      startPolling();
    }
  }

  function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(pollNewAgentMessages, 2500);
  }

  function pollNewAgentMessages() {
    if (!sessionId) return;
    fetch(`${API_BASE}/api/chat/poll?session_id=${encodeURIComponent(sessionId)}`)
      .then(r => r.json())
      .then(data => {
        if (data.new_messages && data.new_messages.length > 0) {
          data.new_messages.forEach(msg => {
            appendMessage(msg.text, "agent");
          });
        }
      })
      .catch(() => {});
  }

  function sendMessage(overrideText) {
    const input = document.getElementById("slw-input");
    const message = (overrideText || input.value).trim();
    if (!message) return;

    appendMessage(message, "user");
    input.value = "";

    fetch(API_BASE + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        session_id: sessionId,
        site_id: SITE_ID
      })
    })
      .then(res => res.json())
      .then(data => {
        if (data.session_id) {
          sessionId = data.session_id;
          localStorage.setItem("slw_session_" + SITE_ID, sessionId);
        }

        if (data.mode === "live") {
          setLiveModeUI(true);
        }

        if (data.reply) {
          appendMessage(data.reply, "bot");
        }
      })
      .catch(() => {
        appendMessage("Sorry, I'm having trouble connecting right now. Please try WhatsApp instead.", "bot");
      });
  }

  // Event wiring
  bubble.addEventListener("click", () => {
    chatWindow.classList.add("slw-open");
    if (isLiveMode) startPolling();
  });

  document.getElementById("slw-close").addEventListener("click", () => {
    chatWindow.classList.remove("slw-open");
  });

  document.getElementById("slw-send").addEventListener("click", () => {
    sendMessage();
  });

  document.getElementById("slw-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });
})();
