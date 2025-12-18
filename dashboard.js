// Dashboard JavaScript for Model Analytics

// API base URL
const API_BASE_URL = window.location.origin;

// Chart instances
let accuracyChart, f1Chart, precisionRecallChart, rocAucChart;

// Load dashboard data on page load
document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
});

// Show loading overlay
function showLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
    }
}

// Hide loading overlay
function hideLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

// Load all dashboard data
async function loadDashboardData() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/models`);
        
        if (!response.ok) {
            throw new Error('Failed to load model data');
        }
        
        const modelsData = await response.json();
        
        // Create statistics cards
        createStatsCards(modelsData);
        
        // Create charts
        createCharts(modelsData);
        
        // Create metrics table
        createMetricsTable(modelsData);
        
        // Create confusion matrices
        createConfusionMatrices(modelsData);
        
    } catch (error) {
        console.error('Error loading dashboard:', error);
        alert('Failed to load dashboard data. Please ensure models are trained.');
    } finally {
        hideLoading();
    }
}

// Create statistics cards
function createStatsCards(modelsData) {
    const statsGrid = document.getElementById('statsGrid');
    
    const modelCount = Object.keys(modelsData).length;
    
    // Find best model
    let bestModel = null;
    let bestF1 = 0;
    
    Object.entries(modelsData).forEach(([name, data]) => {
        const f1 = data.test_metrics.f1_score;
        if (f1 > bestF1) {
            bestF1 = f1;
            bestModel = name;
        }
    });
    
    // Calculate average accuracy
    const avgAccuracy = Object.values(modelsData).reduce((sum, data) => 
        sum + data.test_metrics.accuracy, 0) / modelCount;
    
    // Calculate average F1
    const avgF1 = Object.values(modelsData).reduce((sum, data) => 
        sum + data.test_metrics.f1_score, 0) / modelCount;
    
    const stats = [
        {
            icon: 'fa-robot',
            value: modelCount,
            label: 'ML Models',
            color: '#667eea'
        },
        {
            icon: 'fa-trophy',
            value: bestModel,
            label: 'Best Model',
            color: '#f5576c',
            isText: true
        },
        {
            icon: 'fa-chart-line',
            value: (avgAccuracy * 100).toFixed(2) + '%',
            label: 'Avg Accuracy',
            color: '#4caf50'
        },
        {
            icon: 'fa-star',
            value: (bestF1 * 100).toFixed(2) + '%',
            label: 'Best F1-Score',
            color: '#ff9800'
        }
    ];
    
    statsGrid.innerHTML = stats.map(stat => `
        <div class="stat-card">
            <div class="stat-icon" style="color: ${stat.color};">
                <i class="fas ${stat.icon}"></i>
            </div>
            <div class="stat-value" ${stat.isText ? 'style="font-size: 1.2em;"' : ''}>
                ${stat.value}
            </div>
            <div class="stat-label">${stat.label}</div>
        </div>
    `).join('');
}

// Create all charts
function createCharts(modelsData) {
    const modelNames = Object.keys(modelsData);
    const colors = [
        '#667eea', '#764ba2', '#f093fb', '#f5576c',
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
    ];
    
    // Prepare data
    const accuracy = modelNames.map(name => modelsData[name].test_metrics.accuracy * 100);
    const f1Score = modelNames.map(name => modelsData[name].test_metrics.f1_score * 100);
    const precision = modelNames.map(name => modelsData[name].test_metrics.precision * 100);
    const recall = modelNames.map(name => modelsData[name].test_metrics.recall * 100);
    const rocAuc = modelNames.map(name => modelsData[name].test_metrics.roc_auc * 100);
    
    // Accuracy Chart
    const accuracyCtx = document.getElementById('accuracyChart');
    if (accuracyCtx) {
        accuracyChart = new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'Accuracy (%)',
                    data: accuracy,
                    backgroundColor: colors,
                    borderColor: colors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
    
    // F1-Score Chart
    const f1Ctx = document.getElementById('f1Chart');
    if (f1Ctx) {
        f1Chart = new Chart(f1Ctx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'F1-Score (%)',
                    data: f1Score,
                    backgroundColor: 'rgba(102, 126, 234, 0.7)',
                    borderColor: '#667eea',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
    
    // Precision vs Recall Chart
    const prCtx = document.getElementById('precisionRecallChart');
    if (prCtx) {
        precisionRecallChart = new Chart(prCtx, {
            type: 'radar',
            data: {
                labels: modelNames,
                datasets: [
                    {
                        label: 'Precision',
                        data: precision,
                        backgroundColor: 'rgba(102, 126, 234, 0.2)',
                        borderColor: '#667eea',
                        borderWidth: 2
                    },
                    {
                        label: 'Recall',
                        data: recall,
                        backgroundColor: 'rgba(245, 87, 108, 0.2)',
                        borderColor: '#f5576c',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
    
    // ROC-AUC Chart
    const rocCtx = document.getElementById('rocAucChart');
    if (rocCtx) {
        rocAucChart = new Chart(rocCtx, {
            type: 'line',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'ROC-AUC Score (%)',
                    data: rocAuc,
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    borderColor: '#4caf50',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
}

// Create metrics table
function createMetricsTable(modelsData) {
    const tableContainer = document.getElementById('metricsTable');
    
    const tableHTML = `
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                    <th>CV Score</th>
                </tr>
            </thead>
            <tbody>
                ${Object.entries(modelsData).map(([name, data]) => `
                    <tr>
                        <td><strong>${name}</strong></td>
                        <td>${(data.test_metrics.accuracy * 100).toFixed(2)}%</td>
                        <td>${(data.test_metrics.precision * 100).toFixed(2)}%</td>
                        <td>${(data.test_metrics.recall * 100).toFixed(2)}%</td>
                        <td>${(data.test_metrics.f1_score * 100).toFixed(2)}%</td>
                        <td>${(data.test_metrics.roc_auc * 100).toFixed(2)}%</td>
                        <td>${(data.cv_metrics.mean_cv_score * 100).toFixed(2)}% (±${(data.cv_metrics.std_cv_score * 100).toFixed(2)}%)</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    tableContainer.innerHTML = tableHTML;
}

// Create confusion matrices
function createConfusionMatrices(modelsData) {
    const container = document.getElementById('confusionMatrices');
    
    const matricesHTML = Object.entries(modelsData).map(([name, data]) => {
        const cm = data.test_metrics.confusion_matrix;
        
        return `
            <div class="confusion-matrix">
                <h4>${name}</h4>
                <table class="matrix-table">
                    <tr>
                        <td class="label">TN: ${cm[0][0]}</td>
                        <td class="label">FP: ${cm[0][1]}</td>
                    </tr>
                    <tr>
                        <td class="label">FN: ${cm[1][0]}</td>
                        <td class="label">TP: ${cm[1][1]}</td>
                    </tr>
                </table>
                <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    <p><strong>True Negatives:</strong> ${cm[0][0]}</p>
                    <p><strong>False Positives:</strong> ${cm[0][1]}</p>
                    <p><strong>False Negatives:</strong> ${cm[1][0]}</p>
                    <p><strong>True Positives:</strong> ${cm[1][1]}</p>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = matricesHTML;
}
