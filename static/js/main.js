document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const fileInfo = document.getElementById('fileInfo');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');

    // 드래그 앤 드롭
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        if (!uploadArea.contains(e.relatedTarget)) {
            uploadArea.classList.remove('dragover');
        }
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // 클릭으로 파일 선택
    uploadArea.addEventListener('click', function() {
        imageInput.click();
    });

    // 파일 선택 이벤트
    imageInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // 파일 처리
    function handleFile(file) {
        if (!validateFile(file)) return;

        // 파일 정보 표시
        const fileSize = (file.size / (1024 * 1024)).toFixed(2);
        fileInfo.innerHTML = `
            <div class="file-selected">
                <i class="fas fa-image"></i>
                <div>
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${fileSize}MB</div>
                </div>
                <button type="button" onclick="clearFile()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        uploadArea.classList.add('has-file');
    }

    // 파일 검증
    function validateFile(file) {
        const maxSize = 10 * 1024 * 1024; // 10MB
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
        
        if (file.size > maxSize) {
            showNotification('파일 크기가 10MB를 초과합니다.', 'error');
            return false;
        }
        
        if (!allowedTypes.includes(file.type.toLowerCase())) {
            showNotification('지원하지 않는 파일 형식입니다.', 'error');
            return false;
        }
        
        return true;
    }

    // 폼 제출
    uploadForm.addEventListener('submit', function(e) {
        if (!imageInput.files || imageInput.files.length === 0) {
            e.preventDefault();
            showNotification('파일을 선택해주세요.', 'error');
            return;
        }
        
        // 로딩 상태
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
    });

    // 신뢰도 바 애니메이션
    const confidenceFill = document.querySelector('.confidence-fill');
    if (confidenceFill) {
        const confidence = confidenceFill.getAttribute('data-confidence');
        
        // 초기값 0
        confidenceFill.style.width = '0%';
        
        // 애니메이션 실행
        setTimeout(() => {
            confidenceFill.style.width = confidence + '%';
        }, 500);
    }

    // 결과 섹션 애니메이션
    const resultSection = document.getElementById('resultSection');
    if (resultSection) {
        resultSection.style.opacity = '0';
        resultSection.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            resultSection.style.opacity = '1';
            resultSection.style.transform = 'translateY(0)';
            
            // 결과로 스크롤
            resultSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 300);
    }

    // 알림 표시
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => notification.classList.add('show'), 100);
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
});

// 파일 초기화
function clearFile() {
    document.getElementById('imageInput').value = '';
    document.getElementById('fileInfo').innerHTML = '';
    document.getElementById('uploadArea').classList.remove('has-file');
}

// 분석 초기화
function resetAnalysis() {
    window.location.reload();
}
