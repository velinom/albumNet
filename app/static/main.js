var selectedGenre;

function httpGetAsync(url, callback) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", url, true); // true for asynchronous
    xmlHttp.send(null);
}

function sendAlbumRequest() {
   document.getElementById('result-image').src = '/generate';
}

function onGenreClick(genre) {
    var genreList = document.getElementById('genre-list').getElementsByClassName('selected');
    if (genreList.length > 0) {
        genreList[0].className = 'genre-item';
    }
    document.getElementById(genre).className = 'genre-item selected';
    selectedGenre = genre;
}