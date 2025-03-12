import numpy as np
from itertools import compress

class ImageSegmentsAligner:
    def __init__(self, images_list, page_wise_texts, yolo_model):
        self.images = images_list
        self.page_wise_texts = page_wise_texts
        self.page_wise_heights = {pg_no: np.array(pg_img).shape[0] for pg_no, pg_img in enumerate(images_list)}
        self.main_segment_wise_sub_segments = None
        self.segments = None
        self.model = yolo_model

    @staticmethod
    def get_paragraph(bounding_boxes, x_ths=1.0, y_ths=0.5):
        # create basic attributes
        box_group = []
        for box in bounding_boxes:
            min_x, min_y, max_x, max_y = box
            height = max_y - min_y
            box_group.append(
                ['', min_x, max_x, min_y, max_y, height, None, 0])  # last element indicates group

        # cluster boxes into paragraph
        current_group = 1
        while len([box for box in box_group if box[7] == 0]) > 0:
            box_group0 = [box for box in box_group if box[7] == 0]  # group0 = non-group
            # new group
            if len([box for box in box_group if box[7] == current_group]) == 0:
                box_group0[0][7] = current_group  # assign first box to form new group
            # try to add group
            else:
                current_box_group = [box for box in box_group if box[7] == current_group]
                mean_height = np.mean([box[5] for box in current_box_group])
                min_gx = min([box[1] for box in current_box_group]) - x_ths * mean_height
                max_gx = max([box[2] for box in current_box_group]) + x_ths * mean_height
                min_gy = min([box[3] for box in current_box_group]) - y_ths * mean_height
                max_gy = max([box[4] for box in current_box_group]) + y_ths * mean_height
                add_box = False
                for box in box_group0:
                    same_horizontal_level = (min_gx <= box[1] <= max_gx) or (min_gx <= box[2] <= max_gx)
                    same_vertical_level = (min_gy <= box[3] <= max_gy) or (min_gy <= box[4] <= max_gy)
                    if same_horizontal_level and same_vertical_level:
                        box[7] = current_group
                        add_box = True
                        break
                # cannot add more box, go to next group
                if add_box == False:
                    current_group += 1

        # arrage order in paragraph
        result = []
        for i in set(box[7] for box in box_group):
            current_box_group = [box for box in box_group if box[7] == i]
            min_gx = min([box[1] for box in current_box_group])
            max_gx = max([box[2] for box in current_box_group])
            min_gy = min([box[3] for box in current_box_group])
            max_gy = max([box[4] for box in current_box_group])

            result.append([min_gx, min_gy, max_gx, max_gy])

        return result


    def get_page_details(self, pdf_page_images_3d):
        images_np = [np.array(image) for image in pdf_page_images_3d]
        page_wise_details = []
        for i in range(len(images_np)):
            image = images_np[i]
            page_details = {'page_no': i, 'page_image': image}
            page_wise_details.append(page_details)

        return page_wise_details

    def get_sub_segments(self, b_boxes):
        sub_segments = []
        paragraph_b_boxes = self.get_paragraph(bounding_boxes=b_boxes, x_ths=1.3, y_ths=1.1)
        sub_segments.extend(paragraph_b_boxes)

        return sub_segments

    def remove_inner_segments(self, segments, max_intersect_portion=0.2):
        sorted_segments = list(sorted(segments, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        is_containables = []
        segments_count = len(segments)
        for i in range(segments_count):
            small_segment = sorted_segments[i]
            sx1, sy1, sx2, sy2 = small_segment
            small_segment_area = (sx2 - sx1) * (sy2 - sy1)
            is_containable = True
            for j in range(i+1, segments_count):
                big_segment = sorted_segments[j]
                bx1, by1, bx2, by2 = big_segment
                ix1, iy1, ix2, iy2 = max(sx1, bx1), max(sy1, by1), min(sx2, bx2), min(sy2, by2)
                i_width = ix2 - ix1
                i_height = iy2 - iy1
                i_area = 0
                if i_width > 0 and i_height > 0:
                    i_area = i_width * i_height
                intersecting_portion = i_area / small_segment_area
                if intersecting_portion > max_intersect_portion:
                    is_containable = False
                    break
            is_containables.append(is_containable)
        filtered_segments = list(compress(sorted_segments, is_containables))
        return filtered_segments

    def remove_improper_segments(self, segments):
        segments = list(filter(lambda bbox: bbox[0] < bbox[2] and bbox[1] < bbox[3], segments))
        return segments


    def get_yolo_segments_non_roboflow_model(self, page_details):

        segments = []
        for cur_page_details in page_details:
            image = cur_page_details['page_image']
            pg_no = cur_page_details['page_no']

            boxes = self.model.predict(image)
            cur_image_segments = boxes.tolist()
            cur_image_segments = [list(map(int, segment)) for segment in cur_image_segments]
            cur_image_segments = self.remove_improper_segments(segments=cur_image_segments)
            cur_image_segments = self.remove_inner_segments(segments=cur_image_segments)
            cur_image_segments = list(sorted(cur_image_segments, key=lambda bbox: [bbox[1], bbox[0], bbox[3], bbox[2]]))

            cur_image_segments_with_pg_no = {'page_no': pg_no, 'main_segments': cur_image_segments}
            segments.append(cur_image_segments_with_pg_no)

        return segments

    def detect_segments(self, images):

        segments = []
        pg_wise_sgmt_bboxes = self.model.bulk_predict(images)
        for pg_no, bboxes in enumerate(pg_wise_sgmt_bboxes):
            cur_image_segments = bboxes.tolist()
            cur_image_segments = [list(map(int, segment)) for segment in cur_image_segments]
            cur_image_segments = self.remove_improper_segments(segments=cur_image_segments)
            cur_image_segments = self.remove_inner_segments(segments=cur_image_segments)
            cur_image_segments.sort(key=lambda bbox: [bbox[1], bbox[0], bbox[3], bbox[2]])

            cur_image_segments_with_pg_no = {'page_no': pg_no, 'main_segments': cur_image_segments}
            segments.append(cur_image_segments_with_pg_no)

        return segments

    def extend_main_segment_margin(self, main_segment, img_height, img_width, x_margin, y_margin):
        extended_main_segment = []
        x1, y1, x2, y2 = main_segment
        ex1 = max(0, x1 - x_margin)
        ey1 = max(0, y1 - y_margin)
        ex2 = min(img_width, x2 + x_margin)
        ey2 = min(img_height, y2 + y_margin)
        extended_main_segment = [ex1, ey1, ex2, ey2]
        return extended_main_segment

    def get_main_segments(self):
        page_wise_detected_main_segments = []
        if len(self.page_wise_texts) > 0:
            page_wise_detected_main_segments = self.detect_segments(images=self.images)

        page_wise_main_segments = {}
        for segment_info in page_wise_detected_main_segments:
            page_no = segment_info['page_no']
            cur_page_text_boxes = self.page_wise_texts[page_no]
            main_segments = segment_info['main_segments']
            cur_page_segments = []
            for main_segment in main_segments:
                mx1, my1, mx2, my2 = main_segment
                cur_main_segment_bboxes = []
                cur_main_segment_texts = []
                for text_bbox in cur_page_text_boxes:
                    word = text_bbox['word']
                    b_box = text_bbox['b_box']
                    tx1, ty1, tx2, ty2 = b_box
                    same_horizontal_level = (mx1 <= tx1 <= mx2) or (mx1 <= tx2 <= mx2)
                    same_vertical_level = (my1 <= ty1 <= my2) or (my1 <= ty2 <= my2)
                    if same_horizontal_level and same_vertical_level:
                        cur_main_segment_bboxes.append(b_box)
                        cur_main_segment_texts.append(word)

                img_height, img_width = np.array(self.images[page_no]).shape[:2]
                x_margin = 5
                y_margin = 5

                main_segment = self.extend_main_segment_margin(main_segment=main_segment,
                                                               img_height=img_height,
                                                               img_width=img_width,
                                                               x_margin=x_margin,
                                                               y_margin=y_margin)

                cur_segment = dict()
                cur_segment['main_segment'] = main_segment
                cur_segment['token_bboxes'] = cur_main_segment_bboxes
                cur_segment['token_words'] = cur_main_segment_texts

                cur_page_segments.append(cur_segment)
            if cur_page_segments:
                page_wise_main_segments[page_no] = cur_page_segments

        return page_wise_main_segments


# def load_model():
#     from src.resume_parser.models.yolo.object_detection import ObjectDetection
#     import yolov9
#     model_path = '/Users/elavarasa-11656/Downloads/resume-segments-detection-model-v1.0.pt'
#     model = ObjectDetection()
#     model.model = yolov9.load(model_path)
#     return model
#
# def get_page_wise_texts(images):
#     import easyocr
#     import numpy as np
#     reader = easyocr.Reader(lang_list=['en'])
#     page_wise_texts = {}
#     for pg_no, img in enumerate(images):
#         results = reader.readtext(np.asarray(img))
#         word_bbox_pairs = []
#         for result in results:
#             word = result[1]
#             box = result[0]
#             bbox = [box[0][0], box[0][1], box[2][0], box[2][1]]
#             word_bbox_pairs.append({'word': word, 'b_box': bbox})
#         page_wise_texts[pg_no] = word_bbox_pairs
#     return page_wise_texts
#
#
#
# def main():
#     from pdf2image import convert_from_path
#     FILE_FOLDER = f"/Users/elavarasa-11656/Desktop/works/Resume_Parser/TestSet_copy"
#     FILE_NAME = "JAGAN.pdf"
#     filepath = f"{FILE_FOLDER}/{FILE_NAME}"
#     images = convert_from_path(filepath, use_cropbox=True)
#     images = [image.convert("RGB") for image in images]
#
#     page_wise_texts = get_page_wise_texts(images)
#
#     model = load_model()
#     obj = ImageSegmentsAligner(images, page_wise_texts,model)
#     main_segments = obj.get_main_segments()
#     print()
#
#
# if __name__ == '__main__':
#     main()