import * as THREE from 'three';
//import { api } from '/scripts/ui/api.ts';
import {getRGBValue} from '/extensions/ComfyUI-3D-Pack/js/sharedFunctions.js';

import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

const visualizer = document.getElementById("visualizer");
const container = document.getElementById( 'container' );
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");
const colorPicker = document.getElementById("color-picker");
const downloadButton = document.getElementById("download-button");

const renderer = new THREE.WebGLRenderer( { antialias: true } );
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( window.innerWidth, window.innerHeight );
container.appendChild( renderer.domElement );

const pmremGenerator = new THREE.PMREMGenerator( renderer );

// scene
const scene = new THREE.Scene();
scene.background = new THREE.Color( 0x000000 );
scene.environment = pmremGenerator.fromScene( new RoomEnvironment( renderer ), 0.04 ).texture;

const ambientLight = new THREE.AmbientLight( 0xffffff , 3.0 );

const camera = new THREE.PerspectiveCamera( 40, window.innerWidth / window.innerHeight, 1, 100 );
camera.position.set( 5, 2, 8 );
const pointLight = new THREE.PointLight( 0xffffff, 15 );
camera.add( pointLight );

const controls = new OrbitControls( camera, renderer.domElement );
controls.target.set( 0, 0.5, 0 );
controls.update();
controls.enablePan = true;
controls.enableDamping = true;

// Handle window reseize event
window.onresize = function () {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

};

const clock = new THREE.Clock();

var lastTimestamp = "";
var needUpdate = false;
let mixer;
let currentURL;
var url = location.protocol + '//' + location.host;

downloadButton.addEventListener('click', e => {
    window.open(currentURL, '_blank');
});

function frameUpdate() {

    var filepath = visualizer.getAttribute("filepath");
    var timestamp = visualizer.getAttribute("timestamp");
    if (timestamp == lastTimestamp){
        if (needUpdate){
            controls.update();
            if (mixer !== undefined) {
                const delta = clock.getDelta();
                mixer.update(delta);
            }
            renderer.render( scene, camera );
        }
        requestAnimationFrame( frameUpdate );
    } else {
        needUpdate = false;
        scene.clear();
        progressDialog.open = true;
        lastTimestamp = timestamp;
        main(filepath);
    }

    var color = getRGBValue(colorPicker.value, true);
    if (color[0] != scene.background.r || color[1] != scene.background.g || color[2] != scene.background.b){
        scene.background.setStyle(colorPicker.value);
        renderer.render( scene, camera ); // Force update background color in preview scene
    }
}

const onProgress = function ( xhr ) {
    if ( xhr.lengthComputable ) {
        progressIndicator.value = xhr.loaded / xhr.total * 100;
    }
};
const onError = function ( e ) {
    console.error( e );
};

async function main(filepath="") {
    // Check if file name is valid
    if (/^.+\.[a-zA-Z]+$/.test(filepath)){

        let params = {"filepath": filepath};
        currentURL = url + '/viewfile?' + new URLSearchParams(params);

        var filepathSplit = filepath.split('.');
        var fileExt = filepathSplit.pop().toLowerCase();
        var filepathNoExt = filepathSplit.join(".");

        if (fileExt == "obj"){
            const loader = new OBJLoader();

            var mtlFolderpath = filepath.substring(0, Math.max(filepath.lastIndexOf("/"), filepath.lastIndexOf("\\"))) + "/";
            var mtlFilepath = filepathNoExt.replace(/^.*[\\\/]/, '') + ".mtl";

            const mtlLoader = new MTLLoader();
            mtlLoader.setPath(url + '/viewfile?' + new URLSearchParams({"filepath": mtlFolderpath}));
            mtlLoader.load( mtlFilepath, function ( mtl ) {
                mtl.preload();
                loader.setMaterials( mtl );
            }, onProgress, onError );

            loader.load( currentURL, function ( obj ) {
                obj.scale.setScalar( 5 );
                scene.add( obj );
                obj.traverse(node => {
                    if (node.material && node.material.map == null) {
                        node.material.vertexColors = true;
                    }
                  });

            }, onProgress, onError );

        } else if (fileExt == "glb") {
            const dracoLoader = new DRACOLoader();
            dracoLoader.setDecoderPath( 'https://unpkg.com/three@latest/examples/jsm/libs/draco/gltf/' );
            const loader = new GLTFLoader();
            loader.setDRACOLoader( dracoLoader );

            loader.load( currentURL, function ( gltf ) {
                const model = gltf.scene;
                //model.position.set( 1, 1, 0 );
                model.scale.set( 3, 3, 3 );

                scene.add( model );
                mixer = new THREE.AnimationMixer(model);
                gltf.animations.forEach((clip) => {
                    mixer.clipAction(clip).play();
                });

            }, onProgress, onError );

        } else if (fileExt == "ply") {

        } else {
            throw new Error(`File extension name has to be either .ply or .splat, got .${fileExt}`);
        }

        needUpdate = true;
    }

    scene.add( ambientLight );
    scene.add( camera );

    progressDialog.close();

    frameUpdate();
}

//main("C:/Users/reall/Softwares/ComfyUI_windows_portable/ComfyUI/output/MeshTest/Mesh_01.obj");
main();
