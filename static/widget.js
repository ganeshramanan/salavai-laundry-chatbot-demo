/**
 * Salavai Laundry Chatbot Widget — Embeddable Script
 *
 * Usage: Add this single line before </body> on any website:
 * <script src="https://salavai-laundry-chatbot-demo.onrender.com/widget.js"></script>
 *
 * This script injects a floating chat bubble + window into the host page,
 * and communicates with the hosted backend via CORS-enabled fetch calls.
 * It does NOT depend on or modify any of the host site's existing CSS/JS.
 */
(function () {
  // ---- CONFIG ----
  const API_BASE = window.location.origin.includes("localhost")
    ? "http://localhost:5008"
    : "https://salavai-laundry-chatbot-demo.onrender.com";

  const PRIMARY_COLOR = "#a41e22";
  const DARK_COLOR = "#14324f";

  // ---- INJECT STYLES (scoped with a unique prefix to avoid collisions) ----
  const style = document.createElement("style");
  style.textContent = `
    #slw-bubble {
      position: fixed; bottom: 24px; right: 24px; width: 60px; height: 60px; border-radius: 50%;
      background: ${PRIMARY_COLOR}; color: #fff; display: flex; align-items: center; justify-content: center;
      font-size: 26px; cursor: pointer; box-shadow: 0 4px 20px rgba(164,30,34,0.4); z-index: 999999;
      transition: transform .2s; border: none; font-family: -apple-system, sans-serif;
    }
    #slw-bubble:hover { transform: scale(1.08); }
    #slw-window {
      position: fixed; bottom: 96px; right: 24px; width: 360px; height: 520px; max-height: 80vh;
      background: #fff; border-radius: 16px; box-shadow: 0 8px 40px rgba(0,0,0,0.25);
      display: none; flex-direction: column; overflow: hidden; z-index: 999999;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #slw-window.slw-open { display: flex; }
    #slw-header {
      background: ${DARK_COLOR}; color: #fff; padding: 16px; font-weight: 700; font-size: 14px;
      display: flex; justify-content: space-between; align-items: center;
    }
    #slw-header .slw-sub { font-size: 11px; color: #a8b8c8; font-weight: 400; margin-top: 2px; }
    #slw-close { cursor: pointer; font-size: 18px; opacity: 0.8; }
    #slw-close:hover { opacity: 1; }
    #slw-messages {
      flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px;
      background: #f9fafb;
    }
    .slw-msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 13px; line-height: 1.5; white-space: pre-line; }
    .slw-msg-bot { background: #fff; border: 1px solid #e5e7eb; align-self: flex-start; }
    .slw-msg-user { background: ${PRIMARY_COLOR}; color: #fff; align-self: flex-end; }
    #slw-quick { display: flex; gap: 6px; flex-wrap: wrap; padding: 0 16px 10px; }
    .slw-quick-reply {
      font-size: 11px; border: 1px solid ${PRIMARY_COLOR}; color: ${PRIMARY_COLOR}; background: #fff;
      padding: 6px 10px; border-radius: 14px; cursor: pointer;
    }
    .slw-quick-reply:hover { background: ${PRIMARY_COLOR}; color: #fff; }
    #slw-input-row { display: flex; gap: 8px; padding: 12px; border-top: 1px solid #e5e7eb; }
    #slw-input { flex: 1; border: 1px solid #e5e7eb; border-radius: 20px; padding: 10px 14px; font-size: 13px; }
    #slw-input:focus { outline: none; border-color: ${PRIMARY_COLOR}; }
    #slw-send {
      background: ${PRIMARY_COLOR}; color: #fff; border: none; border-radius: 50%; width: 38px; height: 38px;
      cursor: pointer; font-size: 14px;
    }
  `;
  document.head.appendChild(style);

  // ---- INJECT HTML ----
  const bubble = document.createElement("button");
  bubble.id = "slw-bubble";
  bubble.textContent = "💬";
  document.body.appendChild(bubble);

  const chatWindow = document.createElement("div");
  chatWindow.id = "slw-window";
  chatWindow.innerHTML = `
    <div id="slw-header">
      <div>
        Salavai Laundry Assistant
        <div class="slw-sub">Ask about pricing, pickup, or franchise opportunities</div>
      </div>
      <div id="slw-close">✕</div>
    </div>
    <div id="slw-messages">
      <div class="slw-msg slw-msg-bot">👋 Hi! I can help with pricing, pickup, turnaround time, or becoming a business partner. What would you like to know?</div>
    </div>
    <div id="slw-quick">
      <div class="slw-quick-reply" data-msg="What is KG Laundry pricing?">KG Laundry pricing</div>
      <div class="slw-quick-reply" data-msg="How does pickup work?">Pickup process</div>
      <div class="slw-quick-reply" data-msg="I want to become a franchise partner">Become a partner</div>
    </div>
    <div id="slw-input-row">
      <input type="text" id="slw-input" placeholder="Type your question...">
      <button id="slw-send">➤</button>
    </div>
  `;
  document.body.appendChild(chatWindow);

  // ---- STATE ----
  let sessionState = {};

  // ---- LOGIC ----
  function appendMessage(text, sender) {
    const container = document.getElementById("slw-messages");
    const div = document.createElement("div");
    div.className = "slw-msg " + (sender === "user" ? "slw-msg-user" : "slw-msg-bot");
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
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
      body: JSON.stringify({ message: message, state: sessionState })
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        appendMessage(data.reply, "bot");
        sessionState = data.state || {};
      })
      .catch(function () {
        appendMessage("Sorry, I'm having trouble connecting right now. Please try WhatsApp instead.", "bot");
      });
  }

  // ---- EVENT WIRING ----
  bubble.addEventListener("click", function () {
    chatWindow.classList.add("slw-open");
  });
  document.getElementById("slw-close").addEventListener("click", function () {
    chatWindow.classList.remove("slw-open");
  });
  document.getElementById("slw-send").addEventListener("click", function () {
    sendMessage();
  });
  document.getElementById("slw-input").addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
  });
  document.querySelectorAll(".slw-quick-reply").forEach(function (el) {
    el.addEventListener("click", function () {
      sendMessage(el.getAttribute("data-msg"));
    });
  });
})();
