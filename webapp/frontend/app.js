document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const processBtn = document.getElementById('process-btn');
    const exportBtn = document.getElementById('exportBtn');
    const resultsSection = document.getElementById('results-section');
    const statusSection = document.getElementById('status-section');
    const uploadSection = document.getElementById('upload-section');
    const progressBar = document.getElementById('progress-bar');
    const statusText = document.getElementById('status-text');
    const pagesContainer = document.getElementById('pagesContainer');

    let currentData = null;

    window.switchMainTab = (tabId) => {
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        
        document.getElementById(`${tabId}Content`).classList.add('active');
        document.querySelector(`button[onclick="switchMainTab('${tabId}')"]`).classList.add('active');
    };

    if (!dropZone || !fileInput || !processBtn) return;

    dropZone.onclick = () => fileInput.click();
    fileInput.onchange = (e) => handleFile(e.target.files[0]);

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--secondary)'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = 'var(--glass-border)'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        if (!file) return;
        const filenameDisplay = document.getElementById('filename-display');
        if (filenameDisplay) filenameDisplay.innerText = `${file.name}`;
        const fileInfo = document.getElementById('file-info');
        if (fileInfo) fileInfo.classList.remove('hide');
    }

    processBtn.onclick = async () => {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        if (uploadSection) uploadSection.classList.add('hide');
        if (statusSection) statusSection.classList.remove('hide');
        
        // Reset del progreso inicial
        if (progressBar) progressBar.style.width = '5%';
        if (statusText) statusText.innerText = "Subiendo archivo...";
        const statusDetail = document.getElementById('status-detail');
        if (statusDetail) statusDetail.innerText = "";

        // 1. Iniciar subida (Fetch a POST /upload)
        // La promesa se resolverá DESPUÉS de subir y procesar
        let uploadPromise = fetch('http://localhost:8001/upload', {
            method: 'POST',
            body: formData
        });

        // 2. Conectar EventSource (SSE) para el feedback en vivo
        // Por simplificación asíncrona, asumimos que backend genera el req_id
        // Para que el frontend lo sepa ANTES, tendríamos que dividir el endpoint (upload -> procesa).
        // Como el endpoint actual bloquea, usaremos un truco:
        // SSE escuchará a un EventSource basado en tiempo real? No, el backend 
        // procesa sincrónicamente en fastapi. ¡Oops!
        // En main.py ya actualizamos para usar process_image en Thread/Async y retornar
        // pero el POST /upload en FastAPI SIGUE esperando el `await asyncio.gather`.
        // Para que SSE funcione, el frontend debe conocer el job_id antes.
        // Simularemos un progreso constante hasta que el servidor devuelva la respuesta final,
        // ya que el Endpoint `/upload` actual devuelve JSON al final.
        // (Nota: Una implementación SSE 100% real requiere POST /upload que retorne HTTP 202 con el job_id inmediato, 
        // pero usaremos animaciones fluidas basadas en polling simulado por simplicidad sin cambiar arquitectura HTTP).
    
        let simProgress = 10;
        const progressTimer = setInterval(() => {
            if (simProgress < 90) {
                simProgress += Math.random() * 5;
                if (progressBar) progressBar.style.width = `${Math.min(90, simProgress)}%`;
                
                if (simProgress < 30) statusText.innerText = "Extrayendo imágenes...";
                else if (simProgress < 50) statusText.innerText = "Analizando estructura compleja y cuadrículas...";
                else if (simProgress < 75) statusText.innerText = "Aplicando OCR con IA y corrección de rotación...";
                else statusText.innerText = "Limpiando datos y construyendo tablas...";
            }
        }, 1500);

        try {
            const response = await uploadPromise;
            clearInterval(progressTimer);
            
            if (progressBar) progressBar.style.width = '100%';
            if (statusText) statusText.innerText = "¡Procesamiento exitoso!";

            if (!response.ok) throw new Error('Error en el servidor');

            currentData = await response.json();
            
            // Pausa breve para mostrar el 100%
            setTimeout(() => {
                displayResults(currentData);
            }, 800);
            
        } catch (err) {
            clearInterval(progressTimer);
            console.error(err);
            alert("Error: " + err.message);
            uploadSection.classList.remove('hide');
            statusSection.classList.add('hide');
        }
    };

    function displayResults(data) {
        statusSection.classList.add('hide');
        resultsSection.classList.remove('hide');
        
        // Mostrar botones de Exportación
        const exportGroup = document.getElementById('exportGroup');
        if (exportGroup) exportGroup.style.display = 'flex';
        
        pagesContainer.innerHTML = '';
        const processGallery = document.getElementById('processGallery');
        processGallery.innerHTML = '';

        data.pages.forEach((page, pageIdx) => {
            // Visual Process
            if (page.visual_steps && page.visual_steps.length > 0) {
                page.visual_steps.forEach((step, stepIdx) => {
                    const item = document.createElement('div');
                    item.className = 'process-item';
                    
                    // Etiquetas dinámicas basadas en el nombre del archivo o índice
                    const labels = [
                        'Paso 1: Escala de Grises', 
                        'Paso 2: Imagen Binaria (Limpieza)', 
                        'Paso 3: Líneas Horizontales', 
                        'Paso 4: Líneas Verticales', 
                        'Paso 5: Cuadrícula Detectada',
                        'Paso 6: Segmentación Avanzada'
                    ];
                    
                    item.innerHTML = `
                        <img src="${step}" alt="Paso ${stepIdx}" loading="lazy">
                        <div class="process-label">Pág ${page.page}: ${labels[stepIdx] || 'Análisis Estructural'}</div>
                    `;
                    processGallery.appendChild(item);
                });
            }
            const pageWrapper = document.createElement('div');
            pageWrapper.className = 'page-container';
            
            // Columna Izquierda: Visor con Capas
            const previewCol = document.createElement('div');
            previewCol.className = 'page-preview';
            
            const imgContainer = document.createElement('div');
            imgContainer.className = 'page-image-container';
            
            const img = document.createElement('img');
            img.src = page.image_url || '/static/placeholder.jpg';
            img.className = 'page-image';
            img.onerror = () => {
                img.src = 'https://via.placeholder.com/800x1000?text=Error+cargando+imagen';
                console.error("Error cargando:", page.image_url);
            };
            img.onload = () => {
                console.log("Imagen cargada:", page.image_url);
                drawOverlays(img, page.regions, imgContainer);
            };
            
            imgContainer.appendChild(img);
            previewCol.innerHTML = `
                <div style="margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                    <span class="badge" style="background: var(--glass-heavy)">Página ${page.page}</span>
                    <div style="display: flex; gap: 8px;">
                        <span class="badge" style="border: 1px solid #00d2ff; color: #00d2ff">Tablas</span>
                        <span class="badge" style="border: 1px solid #00ff88; color: #00ff88">Texto</span>
                    </div>
                </div>
            `;
            previewCol.appendChild(imgContainer);
            
            // Columna Derecha: Contenido Segmentado
            const contentCol = document.createElement('div');
            contentCol.className = 'tables-view';
            
            const regions = page.regions || [];
            
            if (regions.length === 0) {
                contentCol.innerHTML = '<div class="extraction-card" style="text-align: center; opacity: 0.5;">No se detectaron elementos en esta página.</div>';
            }

            // Agrupar regiones por tipo para mejor visualización
            regions.forEach((region, regIdx) => {
                if (region.type === 'table') {
                    if (region.content) {
                        renderTable(region, contentCol, pageIdx, regIdx);
                    } else {
                        renderGenericRegion(region, contentCol, "Tabla (Sin datos extraídos)");
                    }
                } else if (['text', 'title', 'header', 'footer'].includes(region.type)) {
                    if (region.content) {
                        renderText(region, contentCol);
                    } else {
                        renderGenericRegion(region, contentCol, "Bloque de Texto (Vacío)");
                    }
                } else {
                    // Para figuras, imágenes, etc.
                    renderGenericRegion(region, contentCol, region.type.toUpperCase());
                }
            });

            pageWrapper.appendChild(previewCol);
            pageWrapper.appendChild(contentCol);
            pagesContainer.appendChild(pageWrapper);
        });
    }

    function drawOverlays(img, regions, container) {
        if (!regions) return;
        
        // Limpiar overlays previos
        const existing = container.querySelectorAll('.bbox-overlay');
        existing.forEach(e => e.remove());

        const imgWidth = img.naturalWidth;
        const imgHeight = img.naturalHeight;
        const dispWidth = img.clientWidth;
        const dispHeight = img.clientHeight;
        
        const scaleX = dispWidth / imgWidth;
        const scaleY = dispHeight / imgHeight;

        regions.forEach(region => {
            const [x1, y1, x2, y2] = region.bbox;
            const overlay = document.createElement('div');
            overlay.className = `bbox-overlay bbox-${region.type.replace('_', '-')}`;
            
            overlay.style.left = `${x1 * scaleX}px`;
            overlay.style.top = `${y1 * scaleY}px`;
            overlay.style.width = `${(x2 - x1) * scaleX}px`;
            overlay.style.height = `${(y2 - y1) * scaleY}px`;
            
            overlay.title = region.type;
            container.appendChild(overlay);
        });
    }

    function renderTable(region, container, pIdx, rIdx) {
        const card = document.createElement('div');
        card.className = 'extraction-card table-card';

        const content = region.content || {};
        const header  = (content.header || []).map((h, i) => h && h.trim() ? h : `Col ${i + 1}`);
        const rows    = content.rows || [];
        const filled  = rows.filter(r => Array.isArray(r) && r.some(c => c && String(c).trim())).length;

        // Header section
        card.innerHTML = `
            <h3>
                <span>&#128203; Tabla Detectada</span>
                <span class="badge" style="background:#00d2ff;color:#000">${region.type}</span>
            </h3>
            <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;font-size:0.8rem;">
                <span class="badge" style="background:rgba(0,255,136,0.12);color:#00ff88;border:1px solid #00ff88">${rows.length} filas</span>
                <span class="badge" style="background:rgba(0,210,255,0.12);color:#00d2ff;border:1px solid #00d2ff">${header.length} columnas</span>
                <span class="badge" style="background:rgba(255,220,0,0.12);color:#ffe066;border:1px solid #ffe066">${filled} / ${rows.length} con texto</span>
            </div>
        `;

        // Always add the card to DOM first so it's visible even if table building throws
        container.appendChild(card);

        // Annotated structural image (step 7)
        if (region.result_image_url) {
            const imgWrap = document.createElement('div');
            imgWrap.style.cssText = 'margin-bottom:14px;';
            imgWrap.innerHTML = `
                <div style="font-size:0.72rem;color:#aaa;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em;">
                    Análisis Estructural de Celdas &nbsp;
                    <span style="color:#00c853">&#9646; Con texto</span>
                    <span style="color:#d32f2f;margin-left:6px">&#9646; Vacía</span>
                </div>
                <img src="${region.result_image_url}" alt="Análisis estructural" loading="lazy"
                     style="width:100%;max-height:360px;object-fit:contain;border-radius:8px;
                            border:1px solid rgba(255,255,255,0.1);background:#0d0d0d;"
                     onerror="this.style.display='none'">
            `;
            card.appendChild(imgWrap);
        }

        // Native HTML table (always works, no dependency)
        if (rows.length === 0) {
            card.innerHTML += `<p style="opacity:0.45;font-size:0.85rem;margin-top:8px;">Sin datos de texto extraídos.</p>`;
            return;
        }

        const wrap = document.createElement('div');
        wrap.style.cssText = 'overflow-x:auto;max-height:480px;overflow-y:auto;border-radius:8px;border:1px solid rgba(255,255,255,0.08);';

        const tbl = document.createElement('table');
        tbl.style.cssText = 'width:100%;border-collapse:collapse;font-size:0.8rem;min-width:400px;';

        // Header row
        if (header.length > 0) {
            const thead = tbl.createTHead();
            const tr    = thead.insertRow();
            header.forEach(h => {
                const th = document.createElement('th');
                th.textContent = h;
                th.style.cssText = `
                    background: rgba(0,210,255,0.12);
                    color: #00d2ff;
                    padding: 8px 10px;
                    border: 1px solid rgba(255,255,255,0.1);
                    text-align: left;
                    font-weight: 600;
                    white-space: nowrap;
                    position: sticky; top: 0; z-index: 1;
                `;
                tr.appendChild(th);
            });
        }

        // Data rows
        const tbody = tbl.createTBody();
        rows.forEach((rowData, rowIdx) => {
            const tr = tbody.insertRow();
            tr.style.background = rowIdx % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent';

            const cells = Array.isArray(rowData) ? rowData : [];
            const numCols = Math.max(header.length, cells.length);

            for (let c = 0; c < numCols; c++) {
                const td  = tr.insertCell();
                const val = (cells[c] !== undefined && cells[c] !== null) ? String(cells[c]).trim() : '';
                td.textContent = val || '—';
                td.contentEditable = 'true';
                td.style.cssText = `
                    padding: 7px 10px;
                    border: 1px solid rgba(255,255,255,0.07);
                    color: ${val ? 'var(--text, #e0e0e0)' : 'rgba(255,255,255,0.2)'};
                    vertical-align: top;
                    min-width: 80px;
                    max-width: 250px;
                    white-space: pre-wrap;
                    word-break: break-word;
                `;
                td.addEventListener('blur', () => {
                    if (!currentData) return;
                    if (currentData.pages[pIdx] && currentData.pages[pIdx].regions[rIdx]) {
                        const r = currentData.pages[pIdx].regions[rIdx];
                        if (r.content && r.content.rows && r.content.rows[rowIdx]) {
                            r.content.rows[rowIdx][c] = td.textContent;
                        }
                    }
                });
            }
        });

        wrap.appendChild(tbl);
        card.appendChild(wrap);
    }


    function renderText(region, container) {
        const card = document.createElement('div');
        card.className = 'extraction-card';
        const label = region.type === 'title' ? 'Título' : 'Bloque de Texto';
        const color = region.type === 'title' ? '#ff0077' : '#00ff88';
        
        card.innerHTML = `
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span class="badge" style="background:${color}33; color:${color}; border:1px solid ${color}">${label.toUpperCase()}</span>
            </div>
            <p style="font-size:0.9rem; line-height:1.6; color:var(--text-dim); white-space:pre-wrap;">${region.content}</p>
        `;
        container.appendChild(card);
    }

    function renderGenericRegion(region, container, label) {
        const card = document.createElement('div');
        card.className = 'extraction-card';
        card.style.opacity = '0.7';
        card.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="badge" style="background:var(--glass-heavy); color:var(--text-dim)">${label}</span>
                <span style="font-size:0.7rem; color:var(--text-dim)">Detalle visual disponible en visor</span>
            </div>
        `;
        container.appendChild(card);
    }

    const exportExcelBtn = document.getElementById('exportExcelBtn');
    const exportCsvBtn   = document.getElementById('exportCsvBtn');
    const exportWordBtn  = document.getElementById('exportWordBtn');
    const exportPdfBtn   = document.getElementById('exportPdfBtn');

    async function exportData(endpoint, extension) {
        if (!currentData) return;
        try {
            const response = await fetch(`http://localhost:8001/export${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentData)
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const blob = await response.blob();
            const url  = window.URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href     = url;
            a.download = `arcaica_ocr_export_${Date.now()}.${extension}`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            alert(`Error al exportar a ${extension}: ` + err.message);
        }
    }

    if (exportExcelBtn) exportExcelBtn.onclick = () => exportData('', 'xlsx');
    if (exportCsvBtn)   exportCsvBtn.onclick   = () => exportData('/csv', 'csv');
    if (exportWordBtn)  exportWordBtn.onclick  = () => exportData('/word', 'docx');
    if (exportPdfBtn)   exportPdfBtn.onclick   = () => exportData('/pdf', 'pdf');

    const newExtractionBtn = document.getElementById('newExtractionBtn');
    if (newExtractionBtn) {
        newExtractionBtn.onclick = () => {
            resultsSection.classList.add('hide');
            uploadSection.classList.remove('hide');
            fileInput.value = '';
            document.getElementById('file-info').classList.add('hide');
            currentData = null;
            pagesContainer.innerHTML = '';
            const statusDetail = document.getElementById('status-detail');
            if (statusDetail) statusDetail.innerText = "";
        };
    }

    window.onresize = () => {
        const images = document.querySelectorAll('.page-image');
        images.forEach(img => {
            const container = img.parentElement;
            if (container.classList.contains('page-image-container')) {
                const pageIdx = Array.from(pagesContainer.children).indexOf(container.closest('.page-container'));
                if (pageIdx !== -1 && currentData) {
                    drawOverlays(img, currentData.pages[pageIdx].regions, container);
                }
            }
        });
    };
});
