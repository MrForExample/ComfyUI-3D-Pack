export function getRGBValue(colorString="rgb(0, 0, 0)", scale01=false) {
    var color = colorString.split(/[\D]+/).filter(Boolean)
    for (let index = 0; index < color.length; index++) {
        color[index] = Number(color[index]);
    }

    if (scale01) {
        for (let index = 0; index < color.length; index++) {
            color[index] = color[index] / 255;
        }
    }

    return color;
}