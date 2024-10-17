import { app } from "/scripts/app.js"

class Visualizer {
    constructor(node, container, visualSrc) {
        this.node = node

        this.iframe = document.createElement('iframe')
        Object.assign(this.iframe, {
            scrolling: "no",
            overflow: "hidden",
        })
        this.iframe.src = "/extensions/ComfyUI-3D-Pack/html/" + visualSrc + ".html"
        container.appendChild(this.iframe)
    }

    updateVisual(filepath) {
        const iframeDocument = this.iframe.contentWindow.document
        const previewScript = iframeDocument.getElementById('visualizer')
        previewScript.setAttribute("filepath", filepath)

        const timestamp = Date.now().toString()
        previewScript.setAttribute("timestamp", timestamp)
    }

    remove() {
        this.container.remove()
    }
}

function createVisualizer(node, inputName, typeName, inputData, app) {
    node.name = inputName

    const widget = {
        type: typeName,
        name: "preview3d",
        callback: () => {},
        draw : function(ctx, node, widgetWidth, widgetY, widgetHeight) {
            const margin = 30
            const top_offset = LiteGraph.NODE_TITLE_HEIGHT+margin
            const visible = app.canvas.ds.scale > 0.5 && this.type === typeName

            const [x, y] = node.getBounding();
            const [left, top] = app.canvasPosToClientPos([x, y]);
            const width = node.width * app.canvas.ds.scale;
            const height = (node.height - top_offset ) * app.canvas.ds.scale;

            Object.assign(this.visualizer.style, {
                left: `${left}px`,
                top: `${top+(top_offset * app.canvas.ds.scale)}px`,
                width: `${width}px`,
                height: `${height}px`,
                position: "absolute",
                overflow: "hidden",
            })

            Object.assign(this.visualizer.children[0].style, {
                transformOrigin: "50% 50%",
                width: '100%',
                height: '100%',
                border: '0 none',
            })

            this.visualizer.hidden = !visible
        },
    }

    const container = document.createElement('div')
    container.id = `Comfy3D_${inputName}`

    node.visualizer = new Visualizer(node, container, typeName)
    widget.visualizer = container
    widget.parent = node

    document.body.appendChild(widget.visualizer)

    node.addCustomWidget(widget)

    node.updateParameters = (params) => {
        node.visualizer.updateVisual(params.filepath)
    }

    // Events for drawing backgound
    node.onDrawBackground = function (ctx) {
        if (!this.flags.collapsed) {
            node.visualizer.iframe.hidden = false
        } else {
            node.visualizer.iframe.hidden = true
        }
    }

    // Make sure visualization iframe is always inside the node when resize the node
    node.onResize = function () {
        let [w, h] = this.size
        if (w <= 600) w = 600
        if (h <= 500) h = 500

        if (w > 600) {
            h = w - 100
        }

        this.size = [w, h]
    }

    // Events for remove nodes
    node.onRemoved = () => {
        for (let w in node.widgets) {
            if (node.widgets[w].visualizer) {
                node.widgets[w].visualizer.remove()
            }
        }
    }


    return {
        widget: widget,
    }
}

function registerVisualizer(nodeType, nodeData, nodeClassName, typeName){
    if (nodeData.name == nodeClassName) {
        console.log("[3D Visualizer] Registering node: " + nodeData.name)

        const onNodeCreated = nodeType.prototype.onNodeCreated

        nodeType.prototype.onNodeCreated = async function() {
            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined

            let Preview3DNode = app.graph._nodes.filter(
                (wi) => wi.type == nodeClassName
            )
            let nodeName = `Preview3DNode_${nodeClassName}`

            console.log(`[Comfy3D] Create: ${nodeName}`)

            const result = await createVisualizer.apply(this, [this, nodeName, typeName, {}, app])

            this.setSize([600, 500])

            return r
        }

        nodeType.prototype.onExecuted = async function(message) {
            if (message?.previews) {
                this.updateParameters(message.previews[0])
            }
        }
    }
}

app.registerExtension({
    name: "Mr.ForExample.Visualizer.GS",

    async init (app) {

    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        registerVisualizer(nodeType, nodeData, "[Comfy3D] Preview 3DGS", "gsVisualizer")
        registerVisualizer(nodeType, nodeData, "[Comfy3D] Preview 3DMesh", "threeVisualizer")
    },
})