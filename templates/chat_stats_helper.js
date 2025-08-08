// Helper to fetch and send event statistics to Gemini from chat.html
function sendStatsToGemini() {
  fetch('/log-stats?ts=' + Date.now())
    .then(res => res.json())
    .then(data => {
      if (data.stats) {
        const summary =
          '[Event Statistics Summary]\n' +
          'Event Types: ' + JSON.stringify(data.stats.type_counts) + '\n' +
          'Persons: ' + JSON.stringify(data.stats.person_counts) + '\n' +
          'Events per Hour: ' + JSON.stringify(data.stats.hour_counts) + '\n';
        // Simulate user sending this as a message
        appendMsg('Sent event statistics to AI assistant.', 'user');
        appendMsg('...', 'gemini');
        fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: summary })
        })
        .then(res => res.json())
        .then(data => {
          const msgDiv = chatBox.querySelector('.msg.gemini:last-child');
          const fullText = data.reply;
          let i = 0;
          const duration = 3000;
          const step = Math.max(1, Math.ceil(fullText.length / (duration / 30)));
          function type() {
            i += step;
            msgDiv.textContent = fullText.slice(0, i);
            if (i < fullText.length) {
              setTimeout(type, 30);
            } else {
              msgDiv.textContent = fullText;
            }
          }
          type();
        })
        .catch(() => {
          chatBox.querySelector('.msg.gemini:last-child').textContent = 'Error: Could not reach Gemini.';
        });
      } else {
        appendMsg('No event statistics available to send.', 'user');
      }
    })
    .catch(() => {
      appendMsg('Failed to fetch event statistics.', 'user');
    });
}
