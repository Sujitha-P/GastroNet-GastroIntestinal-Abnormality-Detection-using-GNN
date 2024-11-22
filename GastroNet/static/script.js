document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    var formData = new FormData();
    formData.append('image', document.getElementById('image').files[0]);

    // Show loading indicator
    document.getElementById('result').innerText = 'Processing...';

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Update the result based on server response
        if (data.prediction) {
            document.getElementById('result').innerText = 'Predicted Class: ' + data.prediction;
        } else {
            document.getElementById('result').innerText = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        document.getElementById('result').innerText = 'Error occurred: ' + error.message;
    });
});
