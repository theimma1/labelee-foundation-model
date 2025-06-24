<script>
    let currentTask = 'similarity';
    let uploadedImage = null;
    let uploadedDataset = null;
    let chartInstance = null;
    const API_BASE_URL = 'http://localhost:5000'; // Adjust to your backend URL

    // Mock tokenizer for client-side validation
    const mockTokenizer = {
        vocab: { "<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4, "<NUM>": 5, "<PUNCT>": 6 },
        vocabSize: 10000,
        buildVocab(texts) {
            texts.forEach(text => {
                text = text.toLowerCase().replace(/\d+/g, '<NUM>').replace(/[^a-z\s]/g, '<PUNCT>');
                const words = text.split(/\s+/);
                let idx = Object.keys(this.vocab).length;
                words.forEach(word => {
                    if (!this.vocab[word] && idx < this.vocabSize) {
                        this.vocab[word] = idx++;
                    }
                });
            });
        },
        encode(text, max_length = 128) {
            text = text.toLowerCase().replace(/\d+/g, '<NUM>').replace(/[^a-z\s]/g, '<PUNCT>');
            let words = ["<CLS>", ...text.split(/\s+/), "<SEP>"];
            let input_ids = words.slice(0, max_length).map(word => this.vocab[word] || this.vocab["<UNK>"]);
            let attention_mask = input_ids.map(() => 1);
            while (input_ids.length < max_length) {
                input_ids.push(this.vocab["<PAD>"]);
                attention_mask.push(0);
            }
            return [input_ids, attention_mask];
        }
    };

    mockTokenizer.buildVocab(['example text for vocabulary building']);

    document.addEventListener('DOMContentLoaded', function() {
        setupEventListeners();
        createParticles();
        simulateModelStats();
    });

    function setupEventListeners() {
        document.querySelectorAll('.task-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                currentTask = this.dataset.task;
                updateTaskDescription();
            });
        });

        const imageUpload = document.getElementById('imageUpload');
        const imageFile = document.getElementById('imageFile');
        imageUpload.addEventListener('click', () => imageFile.click());
        imageUpload.addEventListener('dragover', handleDragOver);
        imageUpload.addEventListener('drop', handleDrop);
        imageFile.addEventListener('change', function(e) {
            handleFile(e.target.files[0]);
        });

        const datasetUpload = document.getElementById('datasetUpload');
        const datasetFile = document.getElementById('datasetFile');
        datasetUpload.addEventListener('click', () => datasetFile.click());
        datasetUpload.addEventListener('dragover', handleDragOver);
        datasetUpload.addEventListener('drop', handleDatasetDrop);
        datasetFile.addEventListener('change', handleDatasetSelect);

        document.getElementById('processBtn').addEventListener('click', processData);
        document.getElementById('trainBtn').addEventListener('click', trainModel);

        document.querySelectorAll('nav a').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({ behavior: 'smooth' });
            });
        });
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0]) {
            handleFile(files[0]);
        } else {
            alert('Please drop a valid image file.');
        }
    }

    function handleDatasetDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleDataset(files[0]);
        }
    }

    function handleFile(file) {
        if (!file || !file.name) {
            alert('Please upload a valid image file.');
            return;
        }

        const validExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        if (!validExtensions.includes(fileExtension) || !file.type.startsWith('image/')) {
            alert('Please upload a valid image file (e.g., JPG, PNG, GIF).');
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            alert('Image file is too large. Please upload an image smaller than 5MB.');
            return;
        }

        uploadedImage = file;
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('imagePreview').innerHTML = `
                <div style="margin-top: 1rem;">
                    <img src="${e.target.result}" style="max-width: 100%; height: auto; border-radius: 10px; max-height: 200px;" alt="Uploaded image preview">
                    <p style="margin-top: 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.9rem;">${file.name}</p>
                </div>
            `;
        };
        reader.onerror = () => alert('Failed to read the image file.');
        reader.readAsDataURL(file);
    }

    function handleDataset(file) {
        if (!file || !file.name.endsWith('.csv') && !file.name.endsWith('.zip')) {
            alert('Please upload a valid CSV or ZIP file.');
            return;
        }

        uploadedDataset = file;
        document.getElementById('datasetPreview').innerHTML = `
            <div style="margin-top: 1rem;">
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">${file.name}</p>
            </div>
        `;
    }

    function handleDatasetSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleDataset(file);
        }
    }

    function updateTaskDescription() {
        const descriptions = {
            'similarity': 'Measures how well the image and text description match each other',
            'classification': 'Classifies the image using textual context and descriptions',
            'retrieval': 'Extracts feature representations for image-text retrieval tasks',
            'reconstruction': 'Reconstructs input features for self-supervised learning'
        };
        console.log(`Current task: ${currentTask} - ${descriptions[currentTask]}`);
    }

    async function processData() {
        const textInput = document.getElementById('textInput').value.trim();
        if (!uploadedImage && !textInput) {
            alert('Please provide either an image or text input');
            return;
        }

        const formData = new FormData();
        formData.append('task', currentTask);
        if (uploadedImage) formData.append('image', uploadedImage);
        if (textInput) formData.append('text', textInput);
        formData.append('user_id', 'demo_user');

        document.getElementById('loading').style.display = 'block';
        document.getElementById('resultsContainer').style.display = 'none';
        document.getElementById('trainingChart').style.display = 'none';
        document.getElementById('processBtn').disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/api/process`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const results = await response.json();
            if (results.error) {
                throw new Error(results.error);
            }
            displayResults(results);
        } catch (error) {
            console.error('Error processing data:', error.message);
            const mockResults = await mockModelProcess(uploadedImage, textInput, currentTask);
            displayResults(mockResults);
            alert(`Failed to process data: ${error.message}. Displaying mock results.`);
        } finally {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('processBtn').disabled = false;
        }
    }

    async function mockModelProcess(image, text, task) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        const mockClasses = ['Dog', 'Cat', 'Car', 'Tree', 'Building'];
        const results = [];

        if (task === 'similarity') {
            const score = Math.random() * 0.4 + 0.5;
            results.push({
                title: 'Similarity Score',
                value: score.toFixed(2),
                description: `The image and text have a similarity score of ${score.toFixed(2)}`,
                confidence: score
            });
        } else if (task === 'classification') {
            const classIdx = Math.floor(Math.random() * mockClasses.length);
            const confidence = Math.random() * 0.4 + 0.5;
            results.push({
                title: 'Classification Result',
                value: mockClasses[classIdx],
                description: `The image is classified as a ${mockClasses[classIdx]} with text context`,
                confidence: confidence
            });
        } else if (task === 'retrieval') {
            results.push({
                title: 'Vision Feature Norm',
                value: (Math.random() * 10).toFixed(2),
                description: 'L2 norm of extracted vision features',
                confidence: Math.random() * 0.4 + 0.5
            });
            results.push({
                title: 'Text Feature Norm',
                value: (Math.random() * 10).toFixed(2),
                description: 'L2 norm of extracted text features',
                confidence: Math.random() * 0.4 + 0.5
            });
        } else if (task === 'reconstruction') {
            results.push({
                title: 'Vision Reconstruction Loss',
                value: (Math.random() * 0.1).toFixed(4),
                description: 'MSE loss for vision feature reconstruction',
                confidence: Math.random() * 0.4 + 0.5
            });
            results.push({
                title: 'Text Reconstruction Loss',
                value: (Math.random() * 0.1).toFixed(4),
                description: 'MSE loss for text feature reconstruction',
                confidence: Math.random() * 0.4 + 0.5
            });
        }

        return results;
    }

    async function trainModel() {
        if (!uploadedDataset) {
            alert('Please upload a training dataset');
            return;
        }

        const formData = new FormData();
        formData.append('dataset', uploadedDataset);
        formData.append('user_id', 'demo_user');
        formData.append('task', currentTask);

        document.getElementById('loading').style.display = 'block';
        document.getElementById('resultsContainer').style.display = 'none';
        document.getElementById('trainingChart').style.display = 'none';
        document.getElementById('trainBtn').disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/api/train`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            if (result.error) {
                throw new Error(result.error);
            }
            displayTrainingResults(result);
        } catch (error) {
            console.error('Error training model:', error.message);
            const mockResult = await mockTrainModel();
            displayTrainingResults(mockResult);
            alert(`Failed to train model: ${error.message}. Displaying mock training results.`);
        } finally {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('trainBtn').disabled = false;
        }
    }

    async function mockTrainModel() {
        await new Promise(resolve => setTimeout(resolve, 2000));
        const epochs = 5;
        const trainingLoss = Array.from({ length: epochs }, () => Math.random() * 0.5 + 0.5).map((v, i) => v * (1 - i / epochs));
        const validationLoss = trainingLoss.map(v => v * (Math.random() * 0.2 + 0.9));
        return {
            status: 'Completed',
            final_loss: trainingLoss[trainingLoss.length - 1],
            epochs: epochs,
            loss_curve: {
                epochs: Array.from({ length: epochs }, (_, i) => i + 1),
                training: trainingLoss,
                validation: validationLoss
            }
        };
    }

    function displayResults(results) {
        document.getElementById('resultsContainer').innerHTML = results.map(result => `
            <div class="result-item">
                <h4 style="color: #4ecdc4; margin-bottom: 0.5rem;">${result.title}</h4>
                <div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem;">${result.value}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">${result.description}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
            </div>
        `).join('');
    }

    function displayTrainingResults(metrics) {
        document.getElementById('resultsContainer').innerHTML = `
            <div class="result-item">
                <h4 style="color: #4ecdc4; margin-bottom: 0.5rem;">Training Results</h4>
                <div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem;">${metrics.status}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Final Loss: ${metrics.final_loss ? metrics.final_loss.toFixed(4) : 'N/A'}<br>
                    Epochs Completed: ${metrics.epochs || 'N/A'}<br>
                    Checkpoint: ${metrics.checkpoint || 'N/A'}
                </div>
            </div>
        `;

        if (metrics.loss_curve) {
            document.getElementById('trainingChart').style.display = 'block';
            if (chartInstance) chartInstance.destroy();
            chartInstance = new Chart(document.getElementById('trainingChart'), {
                type: 'line',
                data: {
                    labels: metrics.loss_curve.epochs.map((_, i) => `Epoch ${i + 1}`),
                    datasets: [
                        {
                            label: 'Training Loss',
                            data: metrics.loss_curve.training,
                            borderColor: '#ff6b6b',
                            backgroundColor: 'rgba(255, 107, 107, 0.2)',
                            fill: true
                        },
                        {
                            label: 'Validation Loss',
                            data: metrics.loss_curve.validation || [],
                            borderColor: '#4ecdc4',
                            backgroundColor: 'rgba(78, 205, 196, 0.2)',
                            fill: true
                        }
                    ].filter(dataset => dataset.data.length > 0)
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, title: { display: true, text: 'Loss' } },
                        x: { title: { display: true, text: 'Epoch' } }
                    }
                }
            });
        }
    }

    function createParticles() {
        const particlesContainer = document.getElementById('particles');
        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 20 + 's';
            particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
            particlesContainer.appendChild(particle);
        }
    }

    function simulateModelStats() {
        const statNumbers = document.querySelectorAll('.stat-number');
        statNumbers.forEach(stat => {
            const originalText = stat.textContent;
            if (!isNaN(originalText)) {
                animateNumber(stat, 0, parseInt(originalText), 2000);
            }
        });
    }

    function animateNumber(element, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const current = Math.floor(start + (end - start) * progress);
            element.textContent = current.toLocaleString();
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        requestAnimationFrame(update);
    }

    document.addEventListener('mousemove', function(e) {
        const components = document.querySelectorAll('.component');
        components.forEach(component => {
            const rect = component.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = (y - centerY) / 10;
                const rotateY = (centerX - x) / 10;
                component.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-5px)`;
            } else {
                component.style.transform = '';
            }
        });
    });
</script>