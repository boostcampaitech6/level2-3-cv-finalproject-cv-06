if (videoInfo.frame_urls === "Nothing") {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('container-xl', 'text-center', 'mt-2', 'py-4');
    const messageHeader = document.createElement('h4');
    messageHeader.classList.add('text-start');
    messageHeader.innerText = '아무런 이상행동이 탐지되지 않았습니다.';
    messageDiv.appendChild(messageHeader);
    screenshotContainer.parentNode.insertBefore(messageDiv, screenshotContainer.nextSibling);
} else {
    videoInfo.frame_urls.forEach(frameInfo => {
        const col = document.createElement('div');
        col.classList.add('col');

        const image = document.createElement('img');
        image.classList.add('img-thumbnail');
        image.src = frameInfo[1];

        const timestampLink = document.createElement('a');
        timestampLink.classList.add('link-underline-light', 'text-reset');
        timestampLink.href = '#';
        timestampLink.innerText = frameInfo[2];

        col.appendChild(image);
        col.appendChild(timestampLink);
        screenshotContainer.appendChild(col);

        // 이미지를 누르면 확대된 이미지와 버튼이 있는 팝업창을 띄우기
        image.onclick = () => showPopup(frameInfo[0]);

        // 타임스탬프를 누르면 비디오가 해당 타임스탬프로 이동
        timestampLink.onclick = () => seekVideo(frameInfo[2]);
    });
}

function showPopup(imageUrl) {
    // 팝업창 가운데 정렬을 위한 스크린 가로, 세로 크기 계산
    const screenWidth = window.screen.width;
    const screenHeight = window.screen.height;

    // 팝업창 크기 조정
    const popupWidth = Math.min(screenWidth * 0.8, 800); // 최대 80% 화면 크기 또는 800px
    const popupHeight = Math.min(screenHeight * 0.8, 600); // 최대 80% 화면 크기 또는 600px

    // 팝업창 가운데 정렬을 위한 위치 계산
    const left = (screenWidth - popupWidth) / 2;
    const top = (screenHeight - popupHeight) / 2;

    // 팝업창 열기
    const popupWindow = window.open(`/album/details/images?frame_id=${imageUrl}`, 'ImagePopup', `width=${popupWidth}, height=${popupHeight}, top=${top}, left=${left}`);
    if (popupWindow) {
        popupWindow.focus();
    }
}

function seekVideo(timestamp) {
    // timestamp 문자열을 시, 분, 초로 분해
    const timeParts = timestamp.split(':');
    
    // 각 부분을 정수로 변환
    const hours = parseInt(timeParts[0], 10);
    const minutes = parseInt(timeParts[1], 10);
    const seconds = parseInt(timeParts[2], 10);

    // 비디오의 currentTime 설정
    const video = document.getElementById('vid');
    video.currentTime = hours * 3600 + minutes * 60 + seconds;
}

var loading = '{{ loading|tojson|safe }}';
if (loading) {
    document.body.classList.add('loading');
}