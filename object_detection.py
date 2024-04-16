import cv2
import torch
import time

def equalize_histogram_color(frame):
    # Convert the image from BGR to YCrCb
    ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Split the image into channels
    y_channel, cr, cb = cv2.split(ycrcb_img)

    # Equalize the histogram of the Y channel
    y_channel_eq = cv2.equalizeHist(y_channel)

    # Merge the equalized Y channel back into the YCrCb image
    ycrcb_img_eq = cv2.merge([y_channel_eq, cr, cb])

    # Convert back to BGR
    equalized_frame = cv2.cvtColor(ycrcb_img_eq, cv2.COLOR_YCrCb2BGR)

    return equalized_frame


def reduce_noise(frame):
    # Apply Gaussian Blur to the frame
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return blurred_frame

def capture_video():
    # Load the model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Create an object to capture video from the webcam
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize counters, timer and confidence threshold
    total_frames = 0
    start_time = time.time()
    object_presence = {'bottle': 0, 'fork': 0, 'book': 0}
    confidence_threshold = 0.4

    # Initialize the list to store the FPS of each processed frame
    fps_list = []

    # Loop to continuously fetch frames from the webcam
    try:
        while True:
            # Keep track of time when frame processing starts
            frame_start_time = time.time()

            # Capture frame-by-frame
            ret, frame = cap.read()

            # Check if frame is read correctly
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Increment total frames
            total_frames += 1

            # Apply preprocessing steps
            frame = equalize_histogram_color(frame)
            frame = reduce_noise(frame)

            # Process the frame through YOLOv5
            # The frame is converted from BGR to RGB and then from HWC layout to CHW layout
            results = model(frame[..., ::-1].transpose(2, 0, 1))

            # Get the data from the results in xyxy format
            results_data = results.xyxy[0]

            # Filter the results based on confidence threshold
            filtered_results = [result for result in results_data if result[4] >= confidence_threshold]

            # Dictionary to check if an object was detected in the frame
            detected_objects = {key: False for key in object_presence}

            # Render the results
            for *box, conf, cls in filtered_results:
                # Get class name for each detection
                class_name = model.names[int(cls)]

                if class_name in ['bottle', 'fork', 'book']:
                    detected_objects[class_name] = True
                    label = f'{class_name} {conf:.2f}'
                    # Draw a rectangle around the object with a blue border of thickness 2
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                    # Put the label text above the rectangle
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)
                    print(f"Detected {class_name} with confidence {conf:.2f}")

                # Update the presence counters
                for obj, detected in detected_objects.items():
                    if detected:
                        object_presence[obj] += 1

            # Calculate and display FPS
            frame_end_time = time.time()
            fps = 1 / (frame_end_time - frame_start_time)
            fps_list.append(fps)
            cv2.putText(frame, f'FPS: {fps:.2f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Webcam Feed', frame)

            # Press 'q' on the keyboard to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        end_time = time.time()

        # Release the capture
        cap.release()
        cv2.destroyAllWindows()

        # Calculate and print statistics
        total_duration = end_time - start_time
        minutes_processed = total_duration / 60
        average_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        print(f"Total number of frames: {total_frames}")
        print(f"Total minutes of video processed: {minutes_processed:.2f}")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Frames with a bottle detected: {object_presence['bottle']}")
        print(f"Frames with a fork detected: {object_presence['fork']}")
        print(f"Frames with a book detected: {object_presence['book']}")

# Run the capture function
capture_video()
