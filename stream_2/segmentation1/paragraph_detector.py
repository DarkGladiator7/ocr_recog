from segmentation1.image_segments_aligner import ImageSegmentsAligner
from segmentation1.object_detection import ObjectDetection
import yolov9
import statistics
import torch
torch.serialization.add_safe_globals(["numpy.core.multiarray._reconstruct"])

def load_model():
    model_path = 'segmentation/resume-segments-detection-model-v1.0.pt'
    model = ObjectDetection()
    model.model = yolov9.load(model_path)
    return model

def find_char_area(p_boxes, p_texts):
    apcs = []
    for bbox, text in zip(p_boxes, p_texts):
        w, h = bbox[2], bbox[3]
        area = w * h
        text_len = len(text)
        avg_apc = area / text_len
        apcs.append(avg_apc)
    avg_apc = statistics.mean(apcs)
    return avg_apc

def get_max_area_box(b_boxes):
    min_px = min([box[0] for box in b_boxes])
    max_px = max([box[2] for box in b_boxes])
    min_py = min([box[1] for box in b_boxes])
    max_py = max([box[3] for box in b_boxes])

    return (min_px, min_py, max_px-min_px, max_py-min_py)

def extract_paragraphs(image, texts_b_boxes):
    images = [image]
    page_wise_texts = {0: texts_b_boxes}
    model = load_model()
    obj = ImageSegmentsAligner(images, page_wise_texts, model)
    main_segments = obj.get_main_segments()[0]
    detected_paragraphs = []
    for main_segment_info in main_segments:
        seg_box = main_segment_info['main_segment']

        paragraph_box = get_max_area_box(main_segment_info['token_bboxes'])
        char_area = find_char_area(p_boxes=main_segment_info['token_bboxes'], p_texts=main_segment_info['token_words'])

        # Do not sort the tokens, keep them in the order detected
        sorted_text_boxes = zip(main_segment_info['token_bboxes'], main_segment_info['token_words'])

        # Join words in the original order
        paragraph_text = ' '.join([text for box, text in sorted_text_boxes])

        detected_paragraphs.append((paragraph_text, paragraph_box, char_area))
    return detected_paragraphs

