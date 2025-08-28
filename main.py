import cv2
import pickle

# --- Parking Slot Dimensions (box size for each slot) ---
width, height = 107, 48

# --- Load previously saved parking slot positions (if any) ---
# CarParkPos is a file where we save clicked slot coordinates
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)   # Load saved positions
except:
    posList = []   # If no file exists, start with an empty list

# --- Mouse Click Function ---
# Left Click  -> Add a new parking slot (draws rectangle at that point)
# Right Click -> Remove the selected slot (if rectangle exists there)
def mouseClick(events, x, y, flags, params):
    global posList
    if events == cv2.EVENT_LBUTTONDOWN:     # Add slot
        posList.append((x, y))
    elif events == cv2.EVENT_RBUTTONDOWN:   # Remove slot
        for i, pos in enumerate(posList):
            x1, y1 = pos
            # If right-click is inside a rectangle, remove that slot
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
                break

    # Save updated slot positions back into CarParkPos file
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)

# --- Function to Check Slots (Free vs Occupied) ---
def checkSpaces(imgPro, img):
    free_spaces = 0

    for pos in posList:
        x, y = pos

        # Crop each parking slot from the processed image
        imgCrop = imgPro[y:y + height, x:x + width]

        # Count non-zero pixels (white pixels after thresholding)
        count = cv2.countNonZero(imgCrop)

        # If the count is LOW, it means slot is EMPTY
        if count < 900:
            color = (0, 255, 0)   # Green = Free
            free_spaces += 1
        else:
            color = (0, 0, 255)   # Red = Occupied

        # Draw rectangle (green/red) around the slot
        cv2.rectangle(img, pos, (x + width, y + height), color, 2)

    # Return free and occupied slot counts
    return free_spaces, len(posList) - free_spaces

# --- MAIN LOOP ---
while True:
    # Load parking lot image
    img = cv2.imread("carParkImg.png")

    if img is None:
        print("Error: carParkImg.png not found! Please check file path.")
        break

    # --- Image Preprocessing (convert image into easier-to-detect form) ---
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)          # Reduce noise
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)   # Binarize
    imgMedian = cv2.medianBlur(imgThreshold, 5)             # Smooth image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1) # Highlight objects

    # --- Count Parking Slots ---
    free, occupied = checkSpaces(imgDilate, img)

    # --- Display free/total slots on image ---
    cv2.putText(img, f'Free: {free} / {len(posList)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the final output
    cv2.imshow("Parking System", img)

    # Allow slot marking with mouse
    cv2.setMouseCallback("Parking System", mouseClick)

    # Exit when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Program finished")
        print(f"Empty spots: {free}")
        print(f"Occupied spots: {occupied}")
        break

cv2.destroyAllWindows()



# important-
# run whole python file if not code didnt work 
# it compare the old image and the one we have select to kow the diff