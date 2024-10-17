import * as SPLAT from 'gsplat';
//import { api } from '/scripts/api.js'
import {getRGBValue} from '/extensions/ComfyUI-3D-Pack/js/sharedFunctions.js';

const visualizer = document.getElementById("visualizer");
const canvas = document.getElementById("canvas");
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");
const colorPicker = document.getElementById("color-picker");

const renderer = new SPLAT.WebGLRenderer(canvas);
const scene = new SPLAT.Scene();
const camera = new SPLAT.Camera();
const controls = new SPLAT.OrbitControls(camera, canvas);
controls.orbitSpeed = 3.0;

// Handle window reseize event
const handleResize = () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
};
handleResize();
window.addEventListener("resize", handleResize);

var lastTimestamp = "";
var needUpdate = false;
let currentURL;
var url = location.protocol + '//' + location.host;

function frameUpdate() {

    var filepath = visualizer.getAttribute("filepath");
    var timestamp = visualizer.getAttribute("timestamp");
    if (timestamp == lastTimestamp){
        if (needUpdate){
            controls.update();
            renderer.render( scene, camera );
        }
        requestAnimationFrame( frameUpdate );
    } else {
        needUpdate = false;
        scene.reset();
        progressDialog.open = true;
        lastTimestamp = timestamp;
        main(filepath);
    }

    var color = getRGBValue(colorPicker.value);
    if (color[0] != renderer.backgroundColor.r || color[1] != renderer.backgroundColor.g || color[2] != renderer.backgroundColor.b){
        renderer.backgroundColor = new SPLAT.Color32(color[0], color[1], color[2], 255);  // It will automatically update background color in preview scene
    }
};

const onProgress = function ( progress ) {
    progressIndicator.value = progress * 100;
};

async function main(filepath="") {
    // Check if file name is valid
    if (/^.+\.[a-zA-Z]+$/.test(filepath)){

        let params = {"filepath": filepath};
        currentURL = url + '/viewfile?' + new URLSearchParams(params);
        var splat = null;

        var fileExt = filepath.split('.').pop().toLowerCase();
        if (fileExt == "ply"){
            splat = await SPLAT.PLYLoader.LoadAsync(currentURL, scene, onProgress);
        } else if (fileExt == "splat") {
            splat = await SPLAT.Loader.LoadAsync(currentURL, scene, onProgress);
        } else {
            throw new Error(`File extension name has to be either .ply or .splat, got .${fileExt}`);
        }

        needUpdate = true;
    }

    progressDialog.close();

    frameUpdate();
}

//main("C:/Users/reall/Softwares/ComfyUI_windows_portable/ComfyUI/output/bonsai.splat");
main();