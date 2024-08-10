$(document).ready(function() {
    $('#search-form').on('submit', function(event) {
        event.preventDefault();

        const description = $('#text-input').val().trim();

        if (description) {
            // Send the description to the Flask backend
            $.ajax({
                url: '/api/search',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ query_text: description }),
                success: function(response) {
                    displayResults(response);
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    console.error('Error during the request:', textStatus, errorThrown);
                }
            });
        }
    });
});

function displayResults(imagePaths) {
    $('#results').empty();
    imagePaths.forEach(function(path) {
        $('#results').append(`
            <div class="col-md-4">
                <div class="card">
                    <img src="${path}" class="card-img-top" alt="Image result">
                    <div class="card-body">
                        <p class="card-text">Search result</p>
                    </div>
                </div>
            </div>
        `);
    });
}
