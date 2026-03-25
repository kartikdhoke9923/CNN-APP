// Wait for page to fully load
window.addEventListener('load', function() {
    console.log("🧠 JavaScript loaded!");
    
    // Get all elements
    const imageInput = document.getElementById('imageInput');
    const previewImg = document.getElementById('preview');
    const predictBtn = document.getElementById('predictBtn');
    const resultsDiv = document.getElementById('results');
    
    console.log("Elements found:", imageInput, previewImg, predictBtn);
    
    // 1. File selection - show preview
    imageInput.addEventListener('change', function(e) {
        console.log("File selected!");
        const file = e.target.files[0];
        
        if (file) {
            console.log("File details:", file.name, file.size, file.type);
            
            // Show image preview
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                previewImg.style.display = 'block';
                predictBtn.disabled = false;
                predictBtn.innerHTML = '🚀 Predict';
                resultsDiv.innerHTML = '';
                console.log("Preview shown!");
            };
            reader.readAsDataURL(file);
        }
    });
    
    // 2. Predict button
    predictBtn.addEventListener('click', function() {
        const file = imageInput.files[0];
        if (!file) {
            alert('⚠️ Please select an image first!');
            return;
        }
        
        console.log("Sending to server...");
        predictBtn.disabled = true;
        predictBtn.innerHTML = '⏳ Predicting...';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send to Flask
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("Response:", data);
            if (data.success) {
                // Show top 5 predictions
                resultsDiv.innerHTML = data.predictions.map(p => 
                    `<div class="prediction">
                        <strong>${p.rank}. ${p.class}</strong><br>
                        <span style="color: #1976d2; font-size: 1.1em;">
                            ${(p.confidence_percent)}% confidence
                        </span>
                    </div>`
                ).join('');
            } else {
                resultsDiv.innerHTML = `<div style="color: red; padding: 20px;">❌ ${data.error}</div>`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultsDiv.innerHTML = '<div style="color: red; padding: 20px;">❌ Network error. Check console.</div>';
        })
        .finally(() => {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '🚀 Predict';
        });
    });
});
