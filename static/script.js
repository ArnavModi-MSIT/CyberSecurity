// Update submitFeedback function
function submitFeedback(feedbackType) {
    const buttons = document.querySelectorAll('.feedback-buttons button');
    buttons.forEach(btn => btn.disabled = true);
    
    const url = document.getElementById('resultUrl').textContent;
    
    fetch('/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            url: url,
            type: feedbackType
        })
    })
    .then(response => {
        if(response.ok) {
            alert('Feedback submitted successfully!');
            // Update UI to show new status
            document.getElementById('resultStatus').textContent = feedbackType;
            document.getElementById('resultStatus').className = feedbackType;
            document.getElementById('resultSource').textContent = 'user feedback';
            document.getElementById('resultConfidence').textContent = "User verified";
        } else {
            alert('Error submitting feedback');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to submit feedback');
    })
    .finally(() => {
        buttons.forEach(btn => btn.disabled = false);
    });
}

document.getElementById('detectForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = document.getElementById('urlInput').value;
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');

    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        });

        const data = await response.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }

        document.getElementById('resultUrl').textContent = data.url;
        document.getElementById('resultStatus').textContent = data.prediction;
        document.getElementById('resultStatus').className = data.prediction;
        document.getElementById('resultSource').textContent = data.source;

        if (data.source === 'database') {
            document.getElementById('resultConfidence').textContent = "Verified entry";
        } else {
            document.getElementById('resultConfidence').textContent = 
                `${Math.round(data.confidence * 100)}%`;
        }

        resultDiv.classList.remove('hidden');
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loadingDiv.classList.add('hidden');
    }
});

document.getElementById('chatForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('chatInput');
    const message = input.value;
    input.value = '';

    const chatHistory = document.getElementById('chatHistory');

    // Add user message
    chatHistory.innerHTML += `
        <div class="message user-message">
            <strong>You:</strong> ${message}
        </div>
    `;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        if(data.error) {
            if(data.error === "Chat service unavailable") {
                alert("Chat feature is currently unavailable. Please try again later.");
                return;
            }
            throw new Error(data.error);
        }

        // Add bot response
        chatHistory.innerHTML += `
            <div class="message bot-message">
                <strong>Assistant:</strong> ${data.answer}
            </div>
        `;

        // Scroll to bottom
        chatHistory.scrollTop = chatHistory.scrollHeight;

    } catch(error) {
        alert(`Error: ${error.message}`);
    }
});