ffmpeg -i "$1" -c:v copy -c:a libmp3lame -q:a 4 "${1}.mp3";
