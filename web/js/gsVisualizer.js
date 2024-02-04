import * as SPLAT from 'gsplat';
import { api } from '/scripts/api.js'

const visualizer = document.getElementById("visualizer");
const canvas = document.getElementById("canvas");
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");

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


var lastFilepath = "";
var needUpdate = false;

function frameUpdate() {

    var filepath = visualizer.getAttribute("filepath");
    if (filepath == lastFilepath){
        if (needUpdate){
            controls.update();
            renderer.render( scene, camera );
        }
        requestAnimationFrame( frameUpdate );
    } else {
        needUpdate = false;
        scene.reset();
        progressDialog.open = true;
        lastFilepath = filepath;
        main(lastFilepath);
    }
};

const onProgress = function ( progress ) {
    progressIndicator.value = progress * 100;
};

async function main(filepath="") {
    // Check if file name is valid
    if (/^.+\.[a-zA-Z]+$/.test(filepath)){

        let params = {"filepath": filepath};
        const url = api.apiURL('/viewfile?' + new URLSearchParams(params));
        var splat = null;

        var fileExt = filepath.split('.').pop().toLowerCase();
        if (fileExt == "ply"){
            splat = await SPLAT.PLYLoader.LoadAsync(url, scene, onProgress);
        } else if (fileExt == "splat") {
            splat = await SPLAT.Loader.LoadAsync(url, scene, onProgress);
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