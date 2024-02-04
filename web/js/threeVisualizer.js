import * as THREE from 'three';
import { api } from '/scripts/api.js'

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

const renderer = new THREE.WebGLRenderer( { antialias: true } );
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( window.innerWidth, window.innerHeight );
container.appendChild( renderer.domElement );

const pmremGenerator = new THREE.PMREMGenerator( renderer );

// scene
const scene = new THREE.Scene();
scene.background = new THREE.Color( 0x000000 );
scene.environment = pmremGenerator.fromScene( new RoomEnvironment( renderer ), 0.04 ).texture;

const ambientLight = new THREE.AmbientLight( 0xffffff );

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
        scene.clear();
        progressDialog.open = true;
        lastFilepath = filepath;
        main(lastFilepath);
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
        const url = api.apiURL('/viewfile?' + new URLSearchParams(params));

        var filepathSplit = filepath.split('.');
        var fileExt = filepathSplit.pop().toLowerCase();
        var filepathNoExt = filepathSplit.join("."); 
        
        if (fileExt == "obj"){
            const loader = new OBJLoader();

            var mtlFolderpath = filepath.substring(0, Math.max(filepath.lastIndexOf("/"), filepath.lastIndexOf("\\"))) + "/";
            var mtlFilepath = filepathNoExt.replace(/^.*[\\\/]/, '') + ".mtl";

            const mtlLoader = new MTLLoader();
            mtlLoader.setPath(api.apiURL('/viewfile?' + new URLSearchParams({"filepath": mtlFolderpath})));
            mtlLoader.load( mtlFilepath, function ( mtl ) {
                mtl.preload();
                loader.setMaterials( mtl );
            }, onProgress, onError );
                
            loader.load( url, function ( obj ) {
                obj.scale.setScalar( 5 );
                scene.add( obj );
                
            }, onProgress, onError );


        } else if (fileExt == "glb") {
            const dracoLoader = new DRACOLoader();
            dracoLoader.setDecoderPath( 'https://unpkg.com/three@latest/examples/jsm/libs/draco/gltf/' );
            const loader = new GLTFLoader();
            loader.setDRACOLoader( dracoLoader );

            loader.load( url, function ( gltf ) {
                const model = gltf.scene;
                model.position.set( 1, 1, 0 );
                model.scale.set( 0.01, 0.01, 0.01 );

                scene.add( model );
            
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