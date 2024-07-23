# -*- coding: utf-8 -*-
import io
import base64
import numpy as np
from PIL import Image


def to_html_frame(content):

    html_frame = f"""
    <html>
      <body>
        {content}
      </body>
    </html>
    """

    return html_frame


def to_single_row_table(caption: str, content: str):

    table_html = f"""
    <table border = "1">
        <caption>{caption}</caption>
        <tr>
            <td>{content}</td>
        </tr>
    </table>
    """

    return table_html


def to_image_embed_tag(image: np.ndarray):

    # Convert np.ndarray to bytes
    img = Image.fromarray(image)
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, "PNG")

    # Encode bytes to base64
    image_base64 = base64.b64encode(raw_bytes.getvalue()).decode("utf-8")

    image_tag = f"""
    <img src="data:image/png;base64,{image_base64}" alt="Embedded Image">
    """

    return image_tag
