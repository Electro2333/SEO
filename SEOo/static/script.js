document.addEventListener('DOMContentLoaded', function() {
    // Load content options
    fetch('/content-options')
        .then(response => response.json())
        .then(data => {
            const contentTypes = data.content_types;
            
            // Populate content type select
            const contentTypeSelect = document.getElementById('contentType');
            Object.entries(contentTypes).forEach(([key, type]) => {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = `${type.icon} ${type.name}`;
                contentTypeSelect.appendChild(option);
            });

            // Set up content type change handler
            contentTypeSelect.addEventListener('change', function() {
                const selectedType = contentTypes[this.value];
                updateDynamicOptions(selectedType);
            });

            // Initial population of dynamic options
            updateDynamicOptions(contentTypes[contentTypeSelect.value]);
        });

    // Update dynamic options based on content type
    function updateDynamicOptions(contentType) {
        // Update audiences
        const audienceSelect = document.getElementById('targetAudience');
        audienceSelect.innerHTML = '';
        contentType.audiences.forEach(audience => {
            const option = document.createElement('option');
            option.value = audience;
            option.textContent = audience.charAt(0).toUpperCase() + audience.slice(1);
            audienceSelect.appendChild(option);
        });

        // Update tones
        const toneSelect = document.getElementById('contentTone');
        toneSelect.innerHTML = '';
        contentType.tones.forEach(tone => {
            const option = document.createElement('option');
            option.value = tone;
            option.textContent = tone.charAt(0).toUpperCase() + tone.slice(1);
            toneSelect.appendChild(option);
        });

        // Update languages
        const languageSelect = document.getElementById('language');
        languageSelect.innerHTML = '';
        contentType.languages.forEach(lang => {
            const option = document.createElement('option');
            option.value = lang;
            option.textContent = lang.toUpperCase();
            languageSelect.appendChild(option);
        });
    }

    // Word count slider
    const wordCountSlider = document.getElementById('wordCount');
    const wordCountValue = document.getElementById('wordCountValue');
    wordCountSlider.addEventListener('input', function() {
        wordCountValue.textContent = this.value;
    });

    // Form submission
    document.getElementById('generateForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const submitButton = this.querySelector('button[type="submit"]');
        const contentDiv = document.getElementById('generatedContent');
        const seoDiv = document.getElementById('seoAnalysis');
        const metadataDiv = document.getElementById('metadataContent');
        
        try {
            submitButton.disabled = true;
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Generating...';
            
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content_type: document.getElementById('contentType').value,
                    target_audience: document.getElementById('targetAudience').value,
                    tone: document.getElementById('contentTone').value,
                    language: document.getElementById('language').value,
                    topic: document.getElementById('topic').value,
                    keywords: document.getElementById('keywords').value.split(',').map(k => k.trim()),
                    word_count: parseInt(document.getElementById('wordCount').value),
                    include_faq: document.getElementById('includeFaq').checked,
                    include_howto: document.getElementById('includeHowto').checked,
                    model: document.getElementById('aiModel').value
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Display content
                contentDiv.innerHTML = `<div class="generated-content">${data.data.content}</div>`;
                
                // Display SEO analysis
                seoDiv.innerHTML = generateSEOAnalysisHTML(data.data.seo_score);
                
                // Display metadata
                metadataDiv.innerHTML = generateMetadataHTML(data.data.metadata);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = 'Generate Content';
        }
    });

    function generateSEOAnalysisHTML(seoData) {
        const overallScore = Math.round(seoData.overall * 100);
        let html = `
            <div class="seo-score-card">
                <h4>SEO Score: ${overallScore}%</h4>
                <div class="progress mb-3">
                    <div class="progress-bar ${getScoreClass(overallScore)}" 
                         role="progressbar" 
                         style="width: ${overallScore}%">
                    </div>
                </div>
                
                <h5>Factor Analysis</h5>
                <div class="seo-details">
        `;

        // Add individual factor scores
        for (const [factor, score] of Object.entries(seoData.factors)) {
            const factorScore = Math.round(score * 100);
            html += `
                <div class="seo-factor mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span>${formatFactorName(factor)}</span>
                        <span class="badge ${getScoreClass(factorScore)}">${factorScore}%</span>
                    </div>
                    <div class="progress mt-2">
                        <div class="progress-bar ${getScoreClass(factorScore)}" 
                             role="progressbar" 
                             style="width: ${factorScore}%">
                        </div>
                    </div>
                </div>
            `;
        }

        // Add recommendations if score is less than optimal
        if (seoData.details && seoData.details.length > 0) {
            html += `
                <h5 class="mt-4">Recommendations</h5>
                <div class="recommendations-list">
            `;

            seoData.details.forEach(detail => {
                html += `
                    <div class="recommendation-item mb-3">
                        <div class="d-flex align-items-center">
                            <span class="priority-badge ${detail.priority}">${detail.priority}</span>
                            <strong class="ms-2">${formatFactorName(detail.factor)}</strong>
                        </div>
                        <ul class="mt-2">
                            ${detail.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                `;
            });

            html += '</div>';
        }

        html += `
            <div class="engagement-features mt-4">
                <h5>Engagement Boosters</h5>
                <button class="btn btn-sm btn-outline-primary me-2" onclick="addContentInteraction()">
                    Add Interactive Quiz
                </button>
                <button class="btn btn-sm btn-outline-success" onclick="generateContentSummary()">
                    Generate Key Takeaways
                </button>
            </div>
        `;

        html += '</div></div>';
        return html;
    }

    function getScoreClass(score) {
        if (score >= 80) return 'bg-success';
        if (score >= 60) return 'bg-warning';
        return 'bg-danger';
    }

    function formatFactorName(factor) {
        return factor
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    function generateMetadataHTML(metadata) {
        return `
            <div class="metadata-content">
                <h4>Generated Metadata</h4>
                <pre><code>${JSON.stringify(metadata, null, 2)}</code></pre>
            </div>
        `;
    }

    // Add this function to update the model configuration
    async function updateModelConfig() {
        try {
            const response = await fetch('/update-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: document.getElementById('aiModel').value
                })
            });
            
            const data = await response.json();
            if (data.status !== 'success') {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error updating model:', error);
        }
    }

    // Add model change handler
    document.getElementById('aiModel').addEventListener('change', updateModelConfig);

    // Add new functions
    function addContentInteraction() {
        const content = document.querySelector('.generated-content');
        const quizHTML = `
            <div class="content-quiz mt-4 p-3 border rounded">
                <h6>Test Your Knowledge</h6>
                <p class="quiz-question">What was the main point discussed in this article?</p>
                <div class="quiz-options">
                    ${['Option 1', 'Option 2', 'Option 3'].map(opt => `
                        <div class="form-check">
                            <input class="form-check-input" type="radio" name="quiz">
                            <label class="form-check-label">${opt}</label>
                        </div>
                    `).join('')}
                </div>
                <button class="btn btn-sm btn-primary mt-2">Submit Answer</button>
            </div>
        `;
        content.insertAdjacentHTML('beforeend', quizHTML);
    }

    function generateContentSummary() {
        const content = document.querySelector('.generated-content').textContent;
        fetch('/generate-summary', {
            method: 'POST',
            body: JSON.stringify({ content })
        })
        .then(response => response.json())
        .then(data => {
            const summaryHTML = `
                <div class="content-summary mt-4 p-3 bg-light rounded">
                    <h6>Key Takeaways</h6>
                    <ul>
                        ${data.summary.map(point => `<li>${point}</li>`).join('')}
                    </ul>
                </div>
            `;
            document.querySelector('.generated-content').insertAdjacentHTML('afterbegin', summaryHTML);
        });
    }

    // Add to generateContent function
    const statusMessages = [
        "🧠 Generating initial draft...",
        "🔍 Analyzing content structure...",
        "📊 Evaluating SEO potential...", 
        "✨ Polishing final version..."
    ];

    function updateStatus(message) {
        const statusDiv = document.createElement('div');
        statusDiv.className = 'thinking-status mb-3 p-2 rounded';
        statusDiv.innerHTML = `
            <div class="spinner-border spinner-border-sm text-primary"></div>
            <span class="ms-2">${message}</span>
        `;
        contentDiv.prepend(statusDiv);
    }

    // Add progress visualization
    function showThinkingProcess() {
        const progress = document.createElement('div');
        progress.className = 'thinking-process mb-3';
        progress.innerHTML = `
            <div class="progress-bar-container bg-light rounded-pill">
                <div class="progress-bar bg-primary rounded-pill" 
                     style="width: 0%; transition: width 0.5s ease">
                </div>
            </div>
        `;
        contentDiv.prepend(progress);
        
        let width = 0;
        const interval = setInterval(() => {
            width += 25;
            progress.querySelector('.progress-bar').style.width = `${width}%`;
            if(width >= 100) clearInterval(interval);
        }, 2000);
    }
}); 