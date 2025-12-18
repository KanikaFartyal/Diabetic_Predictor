// Main JavaScript for Diabetes Prediction System

// API base URL
const API_BASE_URL = window.location.origin;

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const predictAllBtn = document.getElementById('predictAllBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');
const resultsContent = document.getElementById('resultsContent');
const allModelsSection = document.getElementById('allModelsSection');
const allModelsContent = document.getElementById('allModelsContent');

// Event Listeners
if (predictionForm) {
    predictionForm.addEventListener('submit', handlePrediction);
}

if (predictAllBtn) {
    predictAllBtn.addEventListener('click', handlePredictAll);
}

// Show loading overlay
function showLoading() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
    }
}

// Hide loading overlay
function hideLoading() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

// Get form data
function getFormData() {
    const formData = new FormData(predictionForm);
    const data = {};
    
    formData.forEach((value, key) => {
        if (key !== 'model') {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    });
    
    return data;
}

// Handle single model prediction
async function handlePrediction(e) {
    e.preventDefault();
    
    showLoading();
    
    const data = getFormData();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
        // Hide all models section if visible
        if (allModelsSection) {
            allModelsSection.style.display = 'none';
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Please try again.');
    } finally {
        hideLoading();
    }
}

// Handle all models prediction
async function handlePredictAll(e) {
    e.preventDefault();
    
    showLoading();
    
    const data = getFormData();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict_all`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayAllModelsResults(result);
        
        // Hide single result section if visible
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get predictions. Please try again.');
    } finally {
        hideLoading();
    }
}

// Display single model results
function displayResults(result) {
    const riskClass = result.risk_level.toLowerCase().replace(' ', '-');
    
    const html = `
        <div class="result-card">
            <div class="result-header">
                <h3>${result.prediction_label}</h3>
                <span class="risk-badge risk-${riskClass}">${result.risk_level}</span>
            </div>
            
            <div class="prediction-result">
                <div class="result-item">
                    <h4>Prediction</h4>
                    <div class="value" style="font-size: 2em; font-weight: bold; color: ${result.prediction === 1 ? '#e74c3c' : '#27ae60'};">
                        ${result.prediction === 1 ? 'Diabetic' : 'Non-Diabetic'}
                    </div>
                </div>
                <div class="result-item">
                    <h4>Diabetes Probability</h4>
                    <div class="value">${result.probability}%</div>
                </div>
                <div class="result-item">
                    <h4>Risk Level</h4>
                    <div class="value" style="font-size: 1.2em;">${result.risk_level}</div>
                </div>
            </div>
            
            <div class="recommendations">
                <h4><i class="fas fa-lightbulb"></i> Health Recommendations</h4>
                <ul>
                    ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #667eea; margin-bottom: 15px;">
                <i class="fas fa-clipboard-list"></i> Input Summary
            </h4>
            <div class="input-summary" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                ${Object.entries(result.input_data).map(([key, value]) => `
                    <div style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
                        <strong>${formatFieldName(key)}:</strong> ${formatValue(key, value)}
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    resultsContent.innerHTML = html;
    resultsSection.style.display = 'block';
    
    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display all models results
function displayAllModelsResults(result) {
    const consensusClass = result.consensus === 'Diabetic' ? 'high' : 'low';
    
    const html = `
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; text-align: center;">
            <h3 style="font-size: 2em; margin-bottom: 10px;">Consensus Prediction</h3>
            <div style="font-size: 2.5em; font-weight: bold; margin: 15px 0;">
                ${result.consensus}
            </div>
            <div style="font-size: 1.2em; opacity: 0.9;">
                ${result.consensus_percentage}% of models agree
            </div>
        </div>
        
        <div class="models-grid">
            ${Object.entries(result.predictions).map(([modelName, prediction]) => `
                <div class="model-card">
                    <h4><i class="fas fa-robot"></i> ${modelName}</h4>
                    <div style="text-align: center; margin: 15px 0;">
                        <span class="risk-badge risk-${prediction.risk_level.toLowerCase().replace(' ', '-')}" style="font-size: 1.3em; font-weight: bold;">
                            ${prediction.prediction === 1 ? 'Diabetic' : 'Non-Diabetic'}
                        </span>
                    </div>
                    <div class="model-metrics">
                        <div class="metric">
                            <div class="metric-label">Probability</div>
                            <div class="metric-value">${prediction.probability}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Risk Level</div>
                            <div class="metric-value" style="font-size: 0.9em;">${prediction.risk_level}</div>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 10px; margin-top: 25px;">
            <h4 style="color: #667eea; margin-bottom: 15px;">
                <i class="fas fa-info-circle"></i> Model Agreement Analysis
            </h4>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;">
                <p><strong>Total Models:</strong> ${Object.keys(result.predictions).length}</p>
                <p><strong>Diabetic Predictions:</strong> ${Object.values(result.predictions).filter(p => p.prediction === 1).length}</p>
                <p><strong>Non-Diabetic Predictions:</strong> ${Object.values(result.predictions).filter(p => p.prediction === 0).length}</p>
                <p><strong>Consensus:</strong> ${result.consensus_percentage}% agreement on ${result.consensus}</p>
            </div>
        </div>
    `;
    
    allModelsContent.innerHTML = html;
    allModelsSection.style.display = 'block';
    
    // Smooth scroll to results
    allModelsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Format field name for display
function formatFieldName(fieldName) {
    const fieldNames = {
        'Pregnancies': 'Pregnancies',
        'Glucose': 'Glucose',
        'BloodPressure': 'Blood Pressure',
        'SkinThickness': 'Skin Thickness',
        'Insulin': 'Insulin',
        'BMI': 'BMI',
        'DiabetesPedigreeFunction': 'DPF',
        'Age': 'Age'
    };
    return fieldNames[fieldName] || fieldName;
}

// Format value for display
function formatValue(fieldName, value) {
    if (fieldName === 'BMI' || fieldName === 'DiabetesPedigreeFunction') {
        return value.toFixed(2);
    }
    return value.toFixed(0);
}

// Add input validation
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
        const min = parseFloat(this.min);
        const max = parseFloat(this.max);
        const value = parseFloat(this.value);
        
        if (value < min) {
            this.value = min;
        } else if (value > max) {
            this.value = max;
        }
    });
});

// Form reset handler
if (predictionForm) {
    predictionForm.addEventListener('reset', function() {
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
        if (allModelsSection) {
            allModelsSection.style.display = 'none';
        }
    });
}

// Sample data button (for testing)
function fillSampleData() {
    document.getElementById('pregnancies').value = 6;
    document.getElementById('glucose').value = 148;
    document.getElementById('bloodPressure').value = 72;
    document.getElementById('skinThickness').value = 35;
    document.getElementById('insulin').value = 125;
    document.getElementById('bmi').value = 33.6;
    document.getElementById('dpf').value = 0.627;
    document.getElementById('age').value = 50;
}

// Export for testing
window.fillSampleData = fillSampleData;
