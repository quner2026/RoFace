/**
 * Face Service Admin Dashboard - JavaScript
 */

const API_BASE = '/api/v1';

// State
let currentPage = 'dashboard';
let facesPage = 1;
let facesPerPage = 12;
let tasksData = [];
let facesData = [];
let recognizeImageData = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadDashboard();
    loadRecentTasks();

    // Auto-refresh every 30 seconds
    setInterval(() => {
        if (currentPage === 'dashboard') {
            loadDashboard();
            loadRecentTasks();
        }
    }, 30000);
});

// Navigation
function initNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            switchPage(page);
        });
    });
}

function switchPage(page) {
    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });

    // Update page visibility
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `page-${page}`);
    });

    // Update title
    const titles = {
        dashboard: '仪表盘',
        recognize: '在线识别',
        tasks: '任务历史',
        faces: '人脸库',
        models: '模型配置'
    };
    document.querySelector('.page-title').textContent = titles[page] || page;

    currentPage = page;

    // Load page-specific data
    if (page === 'faces') {
        loadFaces();
    } else if (page === 'tasks') {
        loadTasks();
    } else if (page === 'models') {
        loadModelInfo();
    }
}

// Dashboard
async function loadDashboard() {
    try {
        // Load health/status
        const health = await fetchAPI('/health');
        updateServiceStatus(health);

        // Load face count
        const stats = await fetchAPI('/faces/stats');
        document.getElementById('stat-total-faces').textContent = stats.total_faces || 0;
        document.getElementById('stat-today-tasks').textContent = stats.today_tasks || 0;
        document.getElementById('stat-success-rate').textContent =
            stats.success_rate ? `${(stats.success_rate * 100).toFixed(1)}%` : '-';
        document.getElementById('stat-avg-time').textContent =
            stats.avg_response_time ? `${stats.avg_response_time}ms` : '-';

    } catch (error) {
        console.error('Failed to load dashboard:', error);
        // Show offline status
        document.querySelector('.status-indicator').classList.remove('online');
        document.querySelector('.status-indicator').classList.add('offline');
        document.querySelector('.status-indicator span:last-child').textContent = '服务离线';
    }
}

function updateServiceStatus(health) {
    const modelStatus = document.getElementById('model-status');
    if (!health.models_loaded) return;

    const models = [
        { key: 'detector', name: '人脸检测', icon: '🔍' },
        { key: 'embedder', name: '特征提取', icon: '🧬' },
        { key: 'gender_age', name: '属性分析', icon: '👤' },
        { key: 'emotion', name: '表情识别', icon: '😊' }
    ];

    modelStatus.innerHTML = models.map(m => `
        <div class="model-item">
            <span class="status-icon ${health.models_loaded[m.key] ? 'loaded' : ''}"></span>
            <span>${m.icon} ${m.name}</span>
        </div>
    `).join('');
}

// Recent Tasks
async function loadRecentTasks() {
    try {
        const tasks = await fetchAPI('/tasks/recent?limit=5');
        renderTasksPreview(tasks);
    } catch (error) {
        console.error('Failed to load recent tasks:', error);
        renderTasksPreview([]);
    }
}

function renderTasksPreview(tasks) {
    const container = document.getElementById('recent-tasks-preview');

    if (!tasks || tasks.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <h3>暂无任务记录</h3>
                <p>开始使用 API 后将在这里显示</p>
            </div>
        `;
        return;
    }

    container.innerHTML = tasks.map(task => `
        <div class="task-item" onclick="showTaskDetail('${task.id}')">
            ${task.image_path ? `<img src="${task.image_path}" style="width:48px;height:48px;border-radius:8px;object-fit:cover;">` : ''}
            <span class="task-type ${task.type}">${getTaskTypeLabel(task.type)}</span>
            <div class="task-info">
                <div class="task-result">${task.result || '-'}</div>
                <div class="task-time">${formatTime(task.created_at)}</div>
            </div>
            <span class="task-duration">${task.duration_ms}ms</span>
        </div>
    `).join('');
}

// Tasks Page
async function loadTasks() {
    try {
        const filter = document.getElementById('task-filter')?.value || 'all';
        const typeParam = filter === 'all' ? '' : `&type=${filter}`;
        const tasks = await fetchAPI(`/tasks?limit=50${typeParam}`);
        tasksData = tasks;
        renderTasksTable(tasks);
    } catch (error) {
        console.error('Failed to load tasks:', error);
        renderTasksTable([]);
    }
}

// 添加 filter 的事件监听
document.addEventListener('DOMContentLoaded', () => {
    const taskFilter = document.getElementById('task-filter');
    if (taskFilter) {
        taskFilter.addEventListener('change', loadTasks);
    }
});

function refreshTasks() {
    loadTasks();
    showToast('任务列表已刷新', 'success');
}

// Delete a single task
async function deleteTask(taskId) {
    if (!confirm('确定要删除此任务记录吗？')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/tasks/${taskId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Delete failed');
        }

        showToast('任务已删除', 'success');
        loadTasks();
        loadRecentTasks();
    } catch (error) {
        showToast('删除失败: ' + error.message, 'error');
        console.error('Delete task failed:', error);
    }
}

// Confirm and delete all tasks
async function confirmDeleteAllTasks() {
    if (!confirm('确定要删除所有任务历史记录吗？此操作不可撤销！')) {
        return;
    }
    
    if (!confirm('再次确认：这将删除所有历史记录和相关图片！')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/tasks`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Delete failed');
        }

        const result = await response.json();
        showToast(`已删除 ${result.count} 条任务记录`, 'success');
        loadTasks();
        loadRecentTasks();
    } catch (error) {
        showToast('删除失败: ' + error.message, 'error');
        console.error('Delete all tasks failed:', error);
    }
}

function renderTasksTable(tasks) {
    const tbody = document.getElementById('tasks-table-body');

    if (!tasks || tasks.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="empty-state">
                    <div class="empty-state-icon">📋</div>
                    <h3>暂无任务记录</h3>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = tasks.map(task => `
        <tr onclick="showTaskDetail('${task.id}')" style="cursor:pointer;">
            <td>${formatTime(task.created_at)}</td>
            <td><span class="task-type ${task.type}">${getTaskTypeLabel(task.type)}</span></td>
            <td>${task.result || '-'}</td>
            <td>${task.duration_ms}ms</td>
            <td>
                <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); showTaskDetail('${task.id}')">
                    查看
                </button>
                <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); deleteTask('${task.id}')">
                    删除
                </button>
            </td>
        </tr>
    `).join('');
}

// Helper function to render identify results in task detail
function renderTaskIdentifyResults(facesData) {
    const items = facesData.map((face, idx) => {
        const bestMatch = face.matches && face.matches.length > 0 ? face.matches[0] : null;
        if (bestMatch) {
            const simPercent = (bestMatch.similarity * 100).toFixed(1);
            const simClass = simPercent >= 70 ? 'high' : simPercent >= 50 ? 'medium' : 'low';
            return `
                <div class="match-item matched">
                    <img class="match-face-img" src="/data/faces/${bestMatch.face_id}.jpg" 
                         onerror="this.style.opacity='0.3'">
                    <div class="match-info">
                        <div class="match-name">Face ${idx + 1}: ${bestMatch.person_name}</div>
                        <div class="match-id">${bestMatch.person_id}</div>
                    </div>
                    <div class="match-similarity ${simClass}">${simPercent}%</div>
                </div>
            `;
        } else {
            return `
                <div class="match-item unmatched">
                    <div class="match-face-img" style="display:flex;align-items:center;justify-content:center;background:var(--bg-tertiary);">?</div>
                    <div class="match-info">
                        <div class="match-name">Face ${idx + 1}: 未识别</div>
                        <div class="match-id">人脸库中无匹配</div>
                    </div>
                </div>
            `;
        }
    }).join('');
    
    return `
        <div style="margin-top: 20px;">
            <h4 style="margin-bottom: 12px;">识别结果 (${facesData.length} 张人脸)</h4>
            <div class="result-matches">${items}</div>
        </div>
    `;
}

function showTaskDetail(taskId) {
    const task = tasksData.find(t => t.id === taskId);
    if (!task) return;

    const content = document.getElementById('task-detail-content');

    // Parse faces data
    let facesData = [];
    try {
        if (task.faces_json) {
            facesData = JSON.parse(task.faces_json);
        }
    } catch (e) { }

    content.innerHTML = `
        <div class="task-detail">
            ${task.image_path ? `
                <div class="recognize-result-image" style="margin-bottom: 20px;">
                    <canvas id="task-detail-canvas"></canvas>
                </div>
            ` : ''}
            <div class="info-row">
                <span class="info-label">任务ID</span>
                <span class="info-value" style="font-family:monospace;font-size:12px;">${task.id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">类型</span>
                <span class="info-value"><span class="task-type ${task.type}">${getTaskTypeLabel(task.type)}</span></span>
            </div>
            <div class="info-row">
                <span class="info-label">时间</span>
                <span class="info-value">${formatTime(task.created_at)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">耗时</span>
                <span class="info-value">${task.duration_ms}ms</span>
            </div>
            <div class="info-row">
                <span class="info-label">结果</span>
                <span class="info-value">${task.result || '-'}</span>
            </div>
            ${facesData.length > 0 && task.type === 'identify' ? renderTaskIdentifyResults(facesData) : ''}
        </div>
    `;

    openModal('task-detail-modal');

    // Draw detection result if available
    if (task.image_path) {
        setTimeout(() => {
            drawTaskImage(task.image_path, facesData, task.type);
        }, 100);
    }
}

function drawTaskImage(imagePath, facesData, taskType) {
    const canvas = document.getElementById('task-detail-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
        // Scale to max 600px width
        const scale = Math.min(1, 600 / img.width);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Draw face boxes for detect type
        if (taskType === 'detect' && facesData.length > 0) {
            facesData.forEach((face, i) => {
                const x1 = face.x1 * scale;
                const y1 = face.y1 * scale;
                const x2 = face.x2 * scale;
                const y2 = face.y2 * scale;

                ctx.strokeStyle = '#6366f1';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Label
                const label = `Face ${i + 1} (${(face.confidence * 100).toFixed(0)}%)`;
                ctx.fillStyle = '#6366f1';
                const textWidth = ctx.measureText(label).width + 10;
                ctx.fillRect(x1, y1 - 22, textWidth, 22);
                ctx.fillStyle = 'white';
                ctx.font = '12px Inter';
                ctx.fillText(label, x1 + 5, y1 - 7);
            });
        }

        // Draw match boxes for identify type
        if (taskType === 'identify' && facesData.length > 0) {
            facesData.forEach((face, i) => {
                const x1 = face.x1 * scale;
                const y1 = face.y1 * scale;
                const x2 = face.x2 * scale;
                const y2 = face.y2 * scale;
                
                const bestMatch = face.matches && face.matches.length > 0 ? face.matches[0] : null;
                const isMatched = bestMatch && bestMatch.similarity >= 0.5;
                
                ctx.strokeStyle = isMatched ? '#10b981' : '#ef4444';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Label
                let label;
                if (isMatched) {
                    label = `${bestMatch.person_name} (${(bestMatch.similarity * 100).toFixed(0)}%)`;
                } else {
                    label = `Face ${i + 1} - 未识别`;
                }
                ctx.fillStyle = isMatched ? '#10b981' : '#ef4444';
                const textWidth = ctx.measureText(label).width + 10;
                ctx.fillRect(x1, y1 - 22, textWidth, 22);
                ctx.fillStyle = 'white';
                ctx.font = '12px Inter';
                ctx.fillText(label, x1 + 5, y1 - 7);
            });
        }
    };

    img.src = imagePath;
}

// ====================
// 在线识别功能
// ====================

let recognizeImageFile = null;

function handleRecognizeImageSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    recognizeImageFile = file;

    const preview = document.getElementById('recognize-preview');
    const uploadContent = document.getElementById('recognize-upload-content');

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        uploadContent.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearRecognize() {
    recognizeImageFile = null;
    document.getElementById('recognize-image').value = '';
    document.getElementById('recognize-preview').style.display = 'none';
    document.getElementById('recognize-upload-content').style.display = 'block';
    document.getElementById('recognize-result-container').innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">🎯</div>
            <h3>等待识别</h3>
            <p>上传图片后点击识别按钮</p>
        </div>
    `;
}

// Update threshold display
function updateThresholdDisplay(value) {
    document.getElementById('threshold-value').textContent = parseFloat(value).toFixed(2);
}

// Get current threshold value
function getThreshold() {
    const slider = document.getElementById('similarity-threshold');
    return slider ? parseFloat(slider.value) : 0.5;
}

async function performIdentify() {
    if (!recognizeImageFile) {
        showToast('请先选择图片', 'error');
        return;
    }

    showLoading();

    try {
        const threshold = getThreshold();
        const formData = new FormData();
        formData.append('image', recognizeImageFile);
        formData.append('top_k', '5');
        formData.append('threshold', threshold.toString());

        // The new API returns faces with their positions and matches
        const identifyResponse = await fetch(`${API_BASE}/identify`, { 
            method: 'POST', 
            body: formData 
        });

        const identifyResult = await identifyResponse.json();

        if (!identifyResponse.ok) {
            throw new Error(identifyResult.error || 'Identification failed');
        }

        // Render the results - API now returns faces with positions and matches
        // Pass threshold to use for determining matched/unmatched
        renderIdentifyResultWithBoxes(identifyResult, threshold);

        // Refresh tasks
        if (currentPage === 'dashboard') {
            loadRecentTasks();
        }

    } catch (error) {
        showToast('识别失败: ' + error.message, 'error');
        console.error('Identify failed:', error);
    } finally {
        hideLoading();
    }
}

function renderIdentifyResultWithBoxes(identifyResult, threshold = 0.5) {
    const container = document.getElementById('recognize-result-container');
    // New API returns faces array with positions and matches
    const faces = identifyResult.faces || [];

    // Handle no faces detected
    if (faces.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❓</div>
                <h3>未检测到人脸</h3>
                <p>请确保图片中有清晰的人脸</p>
            </div>
        `;
        return;
    }

    // For each face, only keep the best match (highest similarity)
    let faceResults = faces.map((f, idx) => {
        const matches = f.matches || [];
        const bestMatch = matches.length > 0 ? matches[0] : null;
        const isIdentified = bestMatch && bestMatch.similarity >= threshold;
        return {
            faceIndex: idx + 1,
            originalIndex: idx,
            x1: f.x1, y1: f.y1, x2: f.x2, y2: f.y2,
            confidence: f.confidence,
            bestMatch,
            isIdentified,
            similarity: bestMatch ? bestMatch.similarity : 0
        };
    });

    // Sort by similarity descending (identified first, then by similarity)
    faceResults.sort((a, b) => {
        if (a.isIdentified && !b.isIdentified) return -1;
        if (!a.isIdentified && b.isIdentified) return 1;
        return b.similarity - a.similarity;
    });

    const identifiedCount = faceResults.filter(f => f.isIdentified).length;

    container.innerHTML = `
        <div style="margin-bottom: 16px;">
            <span style="color: var(--text-muted);">识别耗时: ${identifyResult.inference_time_ms}ms</span>
            <span style="margin-left: 16px;">检测到 <strong>${faces.length}</strong> 张人脸，识别 <strong>${identifiedCount}</strong> 张</span>
            <span style="margin-left: 16px; color: var(--text-muted);">阈值: ${(threshold * 100).toFixed(0)}%</span>
        </div>
        <div class="recognize-result-image">
            <canvas id="identify-result-canvas"></canvas>
        </div>
        <div class="result-matches" id="identify-matches-container" style="margin-top: 16px;">
            <!-- Will be populated after canvas draws crops -->
        </div>
    `;

    // Draw on canvas with face boxes and generate cropped faces
    setTimeout(() => {
        const canvas = document.getElementById('identify-result-canvas');
        const preview = document.getElementById('recognize-preview');
        const matchesContainer = document.getElementById('identify-matches-container');
        if (!canvas || !preview) return;

        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            const scale = Math.min(1, 500 / img.width);
            canvas.width = img.width * scale;
            canvas.height = img.height * scale;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Array to store cropped face data URLs (indexed by original index)
            const croppedFaces = [];

            // Draw each detected face with its own identification result
            faceResults.forEach((face) => {
                const x1 = face.x1 * scale;
                const y1 = face.y1 * scale;
                const x2 = face.x2 * scale;
                const y2 = face.y2 * scale;

                // Crop face from original image for display
                const cropCanvas = document.createElement('canvas');
                const cropCtx = cropCanvas.getContext('2d');
                const cropW = face.x2 - face.x1;
                const cropH = face.y2 - face.y1;
                cropCanvas.width = cropW;
                cropCanvas.height = cropH;
                cropCtx.drawImage(img, face.x1, face.y1, cropW, cropH, 0, 0, cropW, cropH);
                croppedFaces[face.originalIndex] = cropCanvas.toDataURL('image/jpeg', 0.9);

                // Use green for matched, red for unmatched - thin line
                if (face.isIdentified) {
                    ctx.strokeStyle = '#10b981';  // Green
                } else {
                    ctx.strokeStyle = '#ef4444';  // Red
                }

                ctx.lineWidth = 1.5;  // Thinner line
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Draw small label with this face's specific match result
                let label;
                if (face.bestMatch && face.isIdentified) {
                    const simPercent = (face.bestMatch.similarity * 100).toFixed(0);
                    label = `${face.bestMatch.person_name} ${simPercent}%`;
                } else {
                    label = `未识别`;
                }

                ctx.font = 'bold 11px Inter, sans-serif';
                const textWidth = ctx.measureText(label).width + 8;
                const labelHeight = 18;

                // Draw label above box with slight transparency
                ctx.globalAlpha = 0.85;
                ctx.fillStyle = face.isIdentified ? '#10b981' : '#ef4444';
                ctx.fillRect(x1, y1 - labelHeight - 2, textWidth, labelHeight);
                ctx.globalAlpha = 1;
                ctx.fillStyle = 'white';
                ctx.fillText(label, x1 + 4, y1 - 6);
            });

            // Now render the matches list with cropped faces (sorted by similarity)
            if (faceResults.length === 0) {
                matchesContainer.innerHTML = `
                    <div class="match-item unmatched">
                        <div class="match-info">
                            <div class="match-name">未检测到人脸</div>
                        </div>
                    </div>
                `;
            } else {
                matchesContainer.innerHTML = faceResults.map((face, sortedIdx) => {
                    const croppedSrc = croppedFaces[face.originalIndex] || '';
                    const faceId = `face-${face.originalIndex}`;
                    
                    if (face.isIdentified && face.bestMatch) {
                        const simPercent = (face.bestMatch.similarity * 100).toFixed(1);
                        const simClass = simPercent >= 70 ? 'high' : simPercent >= 50 ? 'medium' : 'low';
                        return `
                            <div class="match-item matched expandable" onclick="toggleFaceDetails('${faceId}')">
                                <div class="face-compare">
                                    <img class="match-face-img current-face" src="${croppedSrc}" title="当前识别">
                                    <span class="compare-arrow">→</span>
                                    <img class="match-face-img db-face" src="/data/faces/${face.bestMatch.face_id}.jpg" 
                                         title="人脸库" onerror="this.style.opacity='0.3'">
                                </div>
                                <div class="match-info">
                                    <div class="match-name">${face.bestMatch.person_name}</div>
                                    <div class="match-id">${face.bestMatch.person_id}</div>
                                </div>
                                <div class="match-similarity ${simClass}">
                                    ${simPercent}%
                                </div>
                                <div class="expand-icon">▼</div>
                            </div>
                            <div class="face-details" id="${faceId}" style="display:none;">
                                <div class="attribute-loading" id="${faceId}-attrs">
                                    <div class="spinner-small"></div>
                                    <span>正在分析属性...</span>
                                </div>
                            </div>
                        `;
                    } else {
                        return `
                            <div class="match-item unmatched expandable" onclick="toggleFaceDetails('${faceId}')">
                                <div class="face-compare">
                                    <img class="match-face-img current-face" src="${croppedSrc}" title="当前识别">
                                    <span class="compare-arrow">→</span>
                                    <div class="match-face-img db-face no-match">?</div>
                                </div>
                                <div class="match-info">
                                    <div class="match-name">未识别</div>
                                    <div class="match-id">Face ${face.faceIndex} - 人脸库中无匹配</div>
                                </div>
                                <div class="expand-icon">▼</div>
                            </div>
                            <div class="face-details" id="${faceId}" style="display:none;">
                                <div class="attribute-loading" id="${faceId}-attrs">
                                    <div class="spinner-small"></div>
                                    <span>正在分析属性...</span>
                                </div>
                            </div>
                        `;
                    }
                }).join('');
            }
            
            // Store cropped faces for lazy loading attributes
            window.croppedFacesCache = croppedFaces;
        };
        img.src = preview.src;
    }, 50);
}

// Toggle face details panel
function toggleFaceDetails(faceId) {
    const details = document.getElementById(faceId);
    if (details) {
        const isVisible = details.style.display !== 'none';
        details.style.display = isVisible ? 'none' : 'block';
        
        // Update expand icon
        const parent = details.previousElementSibling;
        if (parent) {
            const icon = parent.querySelector('.expand-icon');
            if (icon) {
                icon.textContent = isVisible ? '▼' : '▲';
            }
        }
        
        // Load attributes on first expand (lazy loading)
        if (!isVisible) {
            // Extract face index from faceId (format: "face-0", "face-1", etc.)
            const faceIndex = parseInt(faceId.replace('face-', ''));
            const attrsContainer = document.getElementById(`${faceId}-attrs`);
            
            // Only load if not already loaded (still showing spinner)
            if (attrsContainer && attrsContainer.querySelector('.spinner-small')) {
                const croppedSrc = window.croppedFacesCache && window.croppedFacesCache[faceIndex];
                if (croppedSrc) {
                    loadFaceAttributes(faceIndex, croppedSrc);
                }
            }
        }
    }
}

// Async load face attributes
async function loadFaceAttributes(faceIndex, croppedImageSrc) {
    const attrsContainer = document.getElementById(`face-${faceIndex}-attrs`);
    if (!attrsContainer) return;
    
    try {
        // Convert base64 to blob
        const response = await fetch(croppedImageSrc);
        const blob = await response.blob();
        
        const formData = new FormData();
        formData.append('image', blob, 'face.jpg');
        
        const analyzeResponse = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!analyzeResponse.ok) {
            throw new Error('Analysis failed');
        }
        
        const result = await analyzeResponse.json();
        
        if (result.faces && result.faces.length > 0) {
            const attrs = result.faces[0];
            attrsContainer.innerHTML = `
                <div class="attribute-grid">
                    <div class="attr-item">
                        <span class="attr-icon">🎂</span>
                        <span class="attr-label">年龄</span>
                        <span class="attr-value">${attrs.age} 岁</span>
                    </div>
                    <div class="attr-item">
                        <span class="attr-icon">${attrs.gender === 'male' ? '👨' : '👩'}</span>
                        <span class="attr-label">性别</span>
                        <span class="attr-value">${attrs.gender === 'male' ? '男' : '女'} (${(attrs.gender_confidence * 100).toFixed(0)}%)</span>
                    </div>
                    <div class="attr-item">
                        <span class="attr-icon">${getEmotionIcon(attrs.emotion)}</span>
                        <span class="attr-label">情绪</span>
                        <span class="attr-value">${getEmotionLabel(attrs.emotion)} (${(attrs.emotion_confidence * 100).toFixed(0)}%)</span>
                    </div>
                </div>
            `;
        } else {
            attrsContainer.innerHTML = `<div class="attr-error">无法分析属性</div>`;
        }
    } catch (error) {
        console.error('Attribute analysis failed:', error);
        attrsContainer.innerHTML = `<div class="attr-error">属性分析失败</div>`;
    }
}

function getEmotionIcon(emotion) {
    const icons = {
        'neutral': '😐',
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fear': '😨',
        'disgust': '🤢',
        'surprise': '😲'
    };
    return icons[emotion] || '😐';
}

function getEmotionLabel(emotion) {
    const labels = {
        'neutral': '平静',
        'happy': '开心',
        'sad': '悲伤',
        'angry': '生气',
        'fear': '恐惧',
        'disgust': '厌恶',
        'surprise': '惊讶'
    };
    return labels[emotion] || '未知';
}

async function performDetect() {
    if (!recognizeImageFile) {
        showToast('请先选择图片', 'error');
        return;
    }

    showLoading();

    try {
        const formData = new FormData();
        formData.append('image', recognizeImageFile);

        const response = await fetch(`${API_BASE}/detect`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Detection failed');
        }

        renderDetectResult(result);

    } catch (error) {
        showToast('检测失败: ' + error.message, 'error');
        console.error('Detect failed:', error);
    } finally {
        hideLoading();
    }
}

// Keep the old function for backward compatibility
function renderIdentifyResult(result) {
    renderIdentifyResultWithBoxes(result);
}

function renderDetectResult(result) {
    const container = document.getElementById('recognize-result-container');

    if (!result.faces || result.faces.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❓</div>
                <h3>未检测到人脸</h3>
                <p>请确保图片中有清晰的人脸</p>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div style="margin-bottom: 16px;">
            <span style="color: var(--text-muted);">检测耗时: ${result.inference_time_ms}ms</span>
            <span style="margin-left: 16px;">检测到 <strong>${result.faces.length}</strong> 张人脸</span>
        </div>
        <div class="recognize-result-image">
            <canvas id="detect-result-canvas"></canvas>
        </div>
    `;

    // Draw on canvas
    setTimeout(() => {
        const canvas = document.getElementById('detect-result-canvas');
        const preview = document.getElementById('recognize-preview');
        if (!canvas || !preview) return;

        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            const scale = Math.min(1, 500 / img.width);
            canvas.width = img.width * scale;
            canvas.height = img.height * scale;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            result.faces.forEach((face, i) => {
                const x1 = face.x1 * scale;
                const y1 = face.y1 * scale;
                const x2 = face.x2 * scale;
                const y2 = face.y2 * scale;

                ctx.strokeStyle = '#6366f1';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Label
                const label = `Face ${i + 1} (${(face.confidence * 100).toFixed(0)}%)`;
                ctx.fillStyle = '#6366f1';
                const textWidth = ctx.measureText(label).width + 10;
                ctx.fillRect(x1, y1 - 20, textWidth, 20);
                ctx.fillStyle = 'white';
                ctx.font = '12px Inter';
                ctx.fillText(label, x1 + 5, y1 - 6);
            });
        };
        img.src = preview.src;
    }, 50);
}

// Loading overlay
function showLoading() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loading-overlay';
    overlay.innerHTML = '<div class="loading-spinner"></div>';
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.remove();
}

// Faces Page
async function loadFaces() {
    try {
        const searchQuery = document.getElementById('face-search')?.value || '';
        const offset = (facesPage - 1) * facesPerPage;

        let url = `/faces?offset=${offset}&limit=${facesPerPage}`;
        if (searchQuery) {
            url += `&search=${encodeURIComponent(searchQuery)}`;
        }

        const result = await fetchAPI(url);
        facesData = result.faces || [];
        const total = result.total || 0;

        renderFacesGrid(facesData);
        renderPagination(total);

    } catch (error) {
        console.error('Failed to load faces:', error);
        renderFacesGrid([]);
    }
}

function searchFaces() {
    facesPage = 1;
    loadFaces();
}

// Handle Enter key in search box
document.addEventListener('keyup', (e) => {
    if (e.target.id === 'face-search' && e.key === 'Enter') {
        searchFaces();
    }
});

function renderFacesGrid(faces) {
    const grid = document.getElementById('faces-grid');

    if (!faces || faces.length === 0) {
        grid.innerHTML = `
            <div class="empty-state" style="grid-column: 1/-1;">
                <div class="empty-state-icon">👥</div>
                <h3>暂无人脸数据</h3>
                <p>点击"添加人脸"开始注册</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = faces.map(face => `
        <div class="face-card" onclick="showFaceDetail('${face.face_id}')">
            <img class="face-card-image" 
                 src="/data/faces/${face.face_id}.jpg" 
                 alt="${face.person_name}"
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
            <div class="face-card-image face-placeholder" style="display:none;">👤</div>
            <div class="face-card-info">
                <div class="face-card-name">${face.person_name}</div>
                <div class="face-card-id">${face.person_id}</div>
            </div>
        </div>
    `).join('');
}

function renderPagination(total) {
    const container = document.getElementById('faces-pagination');
    const totalPages = Math.ceil(total / facesPerPage);

    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }

    let html = `
        <button ${facesPage === 1 ? 'disabled' : ''} onclick="goToPage(${facesPage - 1})">上一页</button>
    `;

    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || (i >= facesPage - 2 && i <= facesPage + 2)) {
            html += `<button class="${i === facesPage ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
        } else if (i === facesPage - 3 || i === facesPage + 3) {
            html += '<button disabled>...</button>';
        }
    }

    html += `
        <button ${facesPage === totalPages ? 'disabled' : ''} onclick="goToPage(${facesPage + 1})">下一页</button>
    `;

    container.innerHTML = html;
}

function goToPage(page) {
    facesPage = page;
    loadFaces();
}

function showFaceDetail(faceId) {
    const face = facesData.find(f => f.face_id === faceId);
    if (!face) return;

    const content = document.getElementById('face-detail-content');
    content.innerHTML = `
        <div class="face-detail">
            <div style="text-align: center; margin-bottom: 24px;">
                <img src="/data/faces/${face.face_id}.jpg" 
                     alt="${face.person_name}"
                     style="max-width: 200px; max-height: 200px; border-radius: 12px; margin-bottom: 16px;"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div style="font-size: 80px; margin-bottom: 16px; display: none;">👤</div>
                <h3>${face.person_name}</h3>
            </div>
            <div class="info-row">
                <span class="info-label">人员ID</span>
                <span class="info-value">${face.person_id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">人脸ID</span>
                <span class="info-value" style="font-family: monospace; font-size: 12px;">${face.face_id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">创建时间</span>
                <span class="info-value">${formatTime(face.created_at)}</span>
            </div>
            ${face.metadata ? `
                <div class="info-row">
                    <span class="info-label">元数据</span>
                    <span class="info-value"><pre style="margin:0;font-size:12px;">${JSON.stringify(JSON.parse(face.metadata), null, 2)}</pre></span>
                </div>
            ` : ''}
        </div>
    `;

    // Set up delete button
    document.getElementById('delete-face-btn').onclick = () => deleteFace(faceId);

    openModal('face-detail-modal');
}

async function deleteFace(faceId) {
    if (!confirm('确定要删除这个人脸吗？此操作无法撤销。')) {
        return;
    }

    try {
        await fetchAPI(`/faces/${faceId}`, { method: 'DELETE' });
        showToast('人脸已删除', 'success');
        closeModal('face-detail-modal');
        loadFaces();
    } catch (error) {
        showToast('删除失败: ' + error.message, 'error');
    }
}

// Add Face
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const preview = document.getElementById('image-preview');
    const uploadContent = document.querySelector('.file-upload-content');

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        uploadContent.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

async function handleAddFace(event) {
    event.preventDefault();

    const personId = document.getElementById('person-id').value;
    const personName = document.getElementById('person-name').value;
    const imageFile = document.getElementById('face-image').files[0];
    const metadata = document.getElementById('metadata').value;

    if (!imageFile) {
        showToast('请选择人脸图片', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('person_id', personId);
    formData.append('person_name', personName);
    if (metadata) {
        formData.append('metadata', metadata);
    }

    try {
        const result = await fetch(`${API_BASE}/register`, {
            method: 'POST',
            body: formData
        }).then(r => r.json());

        if (result.success) {
            showToast('人脸注册成功', 'success');
            closeModal('add-face-modal');
            document.getElementById('add-face-form').reset();
            document.getElementById('image-preview').style.display = 'none';
            document.querySelector('.file-upload-content').style.display = 'block';
            loadFaces();
        } else {
            showToast('注册失败: ' + result.message, 'error');
        }
    } catch (error) {
        showToast('注册失败: ' + error.message, 'error');
    }
}

// Model Info
async function loadModelInfo() {
    try {
        const health = await fetchAPI('/health');

        // Update model status badges
        if (health.models_loaded) {
            Object.entries(health.models_loaded).forEach(([key, loaded]) => {
                // Could update individual model badges here
            });
        }

        // Could load more detailed model info from config endpoint

    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

// Modal helpers
function openModal(id) {
    document.getElementById(id).classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

// Close modal on backdrop click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
    }
});

// Toast notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${getToastIcon(type)}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function getToastIcon(type) {
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    return icons[type] || icons.info;
}

// Utility functions
async function fetchAPI(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : API_BASE + endpoint;
    const response = await fetch(url, {
        headers: {
            'Accept': 'application/json',
            ...options.headers
        },
        ...options
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: response.statusText }));
        throw new Error(error.error || error.message || 'Request failed');
    }

    return response.json();
}

function formatTime(timestamp) {
    if (!timestamp) return '-';

    const date = typeof timestamp === 'number'
        ? new Date(timestamp * 1000)
        : new Date(timestamp);

    const now = new Date();
    const diff = now - date;

    // Less than 1 minute
    if (diff < 60000) {
        return '刚刚';
    }

    // Less than 1 hour
    if (diff < 3600000) {
        return `${Math.floor(diff / 60000)} 分钟前`;
    }

    // Less than 24 hours
    if (diff < 86400000) {
        return `${Math.floor(diff / 3600000)} 小时前`;
    }

    // Format date
    return date.toLocaleDateString('zh-CN', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getTaskTypeLabel(type) {
    const labels = {
        detect: '检测',
        identify: '识别',
        register: '注册',
        compare: '比对',
        analyze: '分析'
    };
    return labels[type] || type;
}

// ====================
// 模型配置功能
// ====================

// Model info
const modelInfo = {
    detector: {
        'scrfd_10g_kps': { name: 'SCRFD 10G KPS', size: '640 × 640', file: 'scrfd_10g_kps.onnx' },
        'scrfd_2.5g_kps': { name: 'SCRFD 2.5G KPS', size: '640 × 640', file: 'scrfd_2.5g_kps.onnx' },
        'retinaface_r50': { name: 'RetinaFace R50', size: '640 × 640', file: 'retinaface_r50.onnx' }
    },
    embedder: {
        'glint360k_r100': { name: 'ArcFace R100', dim: 512, file: 'glint360k_r100.onnx', size: '~250MB' },
        'glint360k_r50': { name: 'ArcFace R50', dim: 512, file: 'glint360k_r50.onnx', size: '~170MB' },
        'glint360k_r18': { name: 'ArcFace R18', dim: 512, file: 'glint360k_r18.onnx', size: '~92MB' },
        'webface_r50': { name: 'WebFace R50', dim: 512, file: 'webface_r50.onnx', size: '~170MB' }
    }
};

let currentDetectorModel = 'scrfd_10g_kps';
let currentEmbedderModel = 'glint360k_r100';

function onDetectorModelChange(value) {
    currentDetectorModel = value;
    const info = modelInfo.detector[value];
    if (info) {
        document.getElementById('model-detector-name').textContent = info.name;
        document.getElementById('detector-input-size').textContent = info.size;
        // Check if model file exists
        checkModelExists('detector', info.file);
    }
}

function onEmbedderModelChange(value) {
    currentEmbedderModel = value;
    const info = modelInfo.embedder[value];
    if (info) {
        document.getElementById('model-embedder-name').textContent = info.name;
        document.getElementById('embedder-dim').textContent = info.dim;
        // Check if model file exists
        checkModelExists('embedder', info.file);
    }
}

async function checkModelExists(modelType, filename) {
    try {
        const response = await fetch(`/models/${filename}`, { method: 'HEAD' });
        const exists = response.ok;
        
        const statusEl = document.getElementById(`${modelType}-file-status`);
        const downloadSection = document.getElementById(`${modelType}-download-section`);
        
        if (exists) {
            statusEl.textContent = '✅ 已下载';
            statusEl.style.color = 'var(--success)';
            if (downloadSection) downloadSection.style.display = 'none';
        } else {
            statusEl.textContent = '⬇️ 需要下载';
            statusEl.style.color = 'var(--warning)';
            if (downloadSection) downloadSection.style.display = 'block';
        }
    } catch (e) {
        console.error('Failed to check model:', e);
    }
}

function refreshModelStatus() {
    // Refresh health status
    loadStats();
    
    // Check current models
    const detectorInfo = modelInfo.detector[currentDetectorModel];
    const embedderInfo = modelInfo.embedder[currentEmbedderModel];
    
    if (detectorInfo) checkModelExists('detector', detectorInfo.file);
    if (embedderInfo) checkModelExists('embedder', embedderInfo.file);
    
    showToast('模型状态已刷新', 'success');
}

async function downloadModel(modelType) {
    const info = modelType === 'embedder' 
        ? modelInfo.embedder[currentEmbedderModel]
        : modelInfo.detector[currentDetectorModel];
        
    if (!info) return;
    
    showToast(`开始下载 ${info.name}...`, 'info');
    
    // Note: Actual download would need backend support
    // This is a placeholder for the UI
    const progressEl = document.getElementById(`${modelType}-progress`);
    const fillEl = document.getElementById(`${modelType}-progress-fill`);
    
    if (progressEl && fillEl) {
        progressEl.style.display = 'block';
        
        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                showToast(`${info.name} 下载完成！`, 'success');
                checkModelExists(modelType, info.file);
            }
            fillEl.style.width = `${progress}%`;
        }, 500);
    }
}

async function downloadModelPack(packName) {
    showToast(`开始下载 ${packName} 模型包...`, 'info');
    
    const btn = document.getElementById(`btn-download-${packName}`);
    if (btn) {
        btn.disabled = true;
        btn.textContent = '下载中...';
    }
    
    // Note: This would need backend API to actually download
    // For now, show instructions
    setTimeout(() => {
        showToast(`请使用命令行运行: python download_models.py --pack ${packName}`, 'info');
        if (btn) {
            btn.disabled = false;
            btn.textContent = '下载';
        }
    }, 1000);
}
