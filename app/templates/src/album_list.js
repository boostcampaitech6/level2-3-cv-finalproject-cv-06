function redirectToDetails(user_id, upload_id) {
    // URL 생성
    let url = `/album/details?user_id=${user_id}&upload_id=${upload_id}`;

    // 페이지 리디렉션
    window.location.href = url;
}

document.addEventListener('DOMContentLoaded', function() {
    let editButtons = document.querySelectorAll('.edit-btn');
    let deleteButtons = document.querySelectorAll('.delete-btn');

    editButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            let uploadId = this.getAttribute('data-uploadid');
            let origin_name = this.getAttribute('data-name');

            document.getElementById('modifyUploadID').value = uploadId;
            document.getElementById('originName').value = origin_name;
        });
    });


    deleteButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            let uploadId = this.getAttribute('data-uploadid');
            let isRealTime = this.getAttribute('data-is-real-time');

            document.getElementById('deleteUploadID').value = uploadId;
            document.getElementById('isRealTime').value = isRealTime;
        });
    });
});