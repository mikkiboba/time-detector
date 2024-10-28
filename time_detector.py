import cv2
import numpy as np


def resize(img: np.ndarray, longer_side: int = 1000) -> np.ndarray:
    """Resized an image.

    Args:
        img (np.ndarray): Image to resize.

    Returns:
        resized_img (np.ndarray): Resized image.
    """

    height, width, _ = img.shape
    scaling = longer_side / max(height, width)
    resized_img = cv2.resize(img, (int(width * scaling), int(height * scaling)))
    return resized_img

def open_img(file_name: str, conversion_flag: int = cv2.COLOR_BGR2GRAY) -> tuple[np.ndarray, np.ndarray]:
    """Opens image from the filename in the project's `imgs` directory.

    Args:
        file_name (str): Name of the required image file.
        conversion_flag (int, optional): Optional flag to convert the image. Defaults to cv2.COLOR_BGR2GRAY.

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple of the requested image and the coverted one.
    """

    path = f"./projects/clocks/imgs/{file_name}"

    image   = cv2.imread(path)
    image   = resize(image)
    gray    = cv2.cvtColor(image, conversion_flag)

    return image, gray


def imshow(image: np.ndarray, title: str = ""):
    """Shows the image on the screen.

    Args:
        image (np.ndarray): Image to show.
        title (str, optional): Optional title of the window. Defaults to "".
    """

    cv2.imshow(title, image)
    cv2.waitKey(0)


def distance_center(x: int, y: int, center_x: int, center_y: int) -> float:
    return np.sqrt((x - center_x)**2 + (y - center_y)**2)


def distance_lines(line1: np.ndarray, line2: np.ndarray) -> float:
    """Calculates the distance between two lines.

    Args:
        line1 (np.ndarray): First line.
        line2 (np.ndarray): Second line.

    Returns:
        distance (float): Distance between the two lines.
    """
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]

    dir1 = np.array([x2_1 - x1_1, y2_1 - y1_1])
    dir2 = np.array([x2_2 - x1_2, y2_2 - y1_2])

    connection_vector = np.array([x1_2 - x1_1, y1_2 - y1_1])

    distance = np.abs(np.cross(dir1, connection_vector)) / np.linalg.norm(dir2)

    return distance


def find_clock(img: np.ndarray) -> tuple[float, float, float]:
    """Finds a clock in an image.

    Args:
        img (np.ndarray): Input image to find the clock in.

    Returns:
        tuple[float, float, float]: Position of center of the clock (x, y) and the radius of the clock's circumference.
    """

    clock_radius: int = 0
    clock_x: int      = 0
    clock_y: int      = 0

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 400, param1=50, param2=100, minRadius=100, maxRadius=500)

    if circles is None:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_rect = None

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > max_area:
                max_area = area
                max_rect = contour
        
        if max_rect is not None:
            (x, y, w, h) = cv2.boundingRect(max_rect)

            clock_x = x + w // 2
            clock_y = y + h // 2

            clock_radius = min(w, h) // 2
    else:
        max_circle: tuple = None

        for circle in circles[0, :]:
            if circle[2] > clock_radius:
                max_circle = circle

            clock_x      = int(max_circle[0])
            clock_y      = int(max_circle[1])
            clock_radius = int(max_circle[2])

    return clock_x, clock_y, clock_radius


def find_lines(img: np.ndarray) -> np.ndarray:
    """Uses Canny edge detection (+ HoughLinesP) to find straight lines in an image.

    Args:
        img (np.ndarray): Input image

    Returns:
        lines (np.ndarray): Lines found in the input image. A line is composed of 4 elements: x1, y1, x2, y2
    """

    edges = cv2.Canny(img, 50 , 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90, minLineLength=30, maxLineGap=5)
    return lines


def find_group_lines(lines: np.ndarray, clock_x: int, clock_y: int, clock_radius: int) -> list[dict]:
    """Groups lines which are parallel and close together

    Args:
        lines (np.ndarray): Lines to group
        clock_x (int): Position X of the center of the clock in the image.
        clock_y (int): Position Y of the center of the clock in the image.
        clock_radius (int): Radius of the clock in the image.

    Returns:
        groups (list[dict]): List with the lines grouped.
    """

    # If the difference between the line angle and the mean angle of the group is less or equals to this, the line falls in that group. 
    group_degrees = 12

    groups = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculating the length from the center of the clock to the endpoints of the line
        len1 = distance_center(x1, y1, clock_x, clock_y)
        len2 = distance_center(x2, y2, clock_x, clock_y)

        min_len = np.min([len1, len2])
        max_len = np.max([len1, len2])
        
        # The farthest point must be within the radius of the clock 
        # The nearest point must be within 50% of the radius of the clock
        if ((max_len < clock_radius) and (min_len < clock_radius*50/100)):

            # Calculating the angle in degrees
            angle = np.arctan2(y2 - y1, x2 - x1)
            angle = np.degrees(angle)

            grouped = False # to check if the line belongs to a group or not
            for group in groups:
                mean_angle = group['mean_angle']
                if abs(angle - mean_angle) < group_degrees or abs(angle - mean_angle - 180) < group_degrees or abs(angle - mean_angle + 180) < group_degrees:
                    group['lines'].append(line)
                    grouped = True
                    break
            if not grouped:
                groups.append({
                    'lines': [line],
                    'mean_angle': angle
                })
    return groups


def find_hands(groups: list[dict], clock_x: int, clock_y: int) -> list[tuple]:
    """Finds the (up to three) hands of the clock.

    Args:
        groups (list[dict]): Grouped lines.
        clock_x (int): Position X of the center of the clock.
        clock_y (int): Position Y of the center of the clock.

    Returns:
        list[tuple]: Hands found in the image.
    """
    
    hands = []

    for group in groups:
        lines = group['lines']
        num_lines = len(lines)

        max_thickness = 0
        max_length = 0

        for i in range(num_lines):
            x1, y1, x2, y2 = lines[i][0]

            # Calculating the distance from the center of the clock
            len1 = distance_center(x1, y1, clock_x, clock_y)
            len2 = distance_center(x2, y2, clock_x, clock_y)

            length = np.max([len1, len2])
            if length > max_length:
                max_length = length

                # The farthest point from the center becomes the end point of the hand
                if length == len1:
                    max_line = x1, y1, clock_x, clock_y
                else:
                    max_line = x2, y2, clock_x, clock_y

            for j in range(i+1, num_lines):
                thickness = distance_lines(lines[i], lines[j])
                if thickness > max_thickness:
                    max_thickness = thickness
        line = max_line, max_thickness, max_length

        # If it's > 0, there are at least two parallel lines
        if max_thickness > 0:
            hands.append(line)

    # Sorting by length in descending order
    hands.sort(key=lambda x: x[2], reverse=True)

    # Taking the first three clock hands
    hands = hands[:min(3, len(hands))]
    return hands


def sort_hands(hands: list[tuple]) -> tuple[tuple | None]:
    """Separates the hour, minute and second hands.

    Args:
        hands (list[tuple]): List of the hands.

    Returns:
        tuple[tuple]: In order, hour, minute and second hands.
    """

    second_hand: tuple = None

    if len(hands) == 3:
        hands_by_thickness = sorted(hands, key=lambda hands: hands[1])
        second_hand = hands_by_thickness[0]
        hands.remove(second_hand)
    
    hands_by_length = sorted(hands, key=lambda hands: hands[2])
    
    hour_hand: tuple   = hands_by_length[0]
    minute_hand: tuple = hands_by_length[1]

    return hour_hand, minute_hand, second_hand


def get_angle(hand: tuple, clock_y: int) -> float:
    """Returns the angle's degrees of the hand.

    Args:
        hand (tuple): The hand to calculate the angle from.
        clock_y (int): Position Y of the clock.

    Returns:
        degrees (float): Degrees of the angle of the hand.
    """

    # Vector direction for the hands
    x1, y1, x2, y2  = hand[0]
    hands_direction = [x2 - x1, y2 - y1]

    # Vector horizontal direction from the center of the clock
    hor_direction = [0, clock_y - (clock_y - 100)]

    # Dot product of the vectors
    dot_dir = hands_direction[0] * hor_direction[0] + hands_direction[1] * hor_direction[1]

    # Vector lengths
    len_hands_dir = np.sqrt(hands_direction[0]**2 + hands_direction[1]**2)
    len_hor_dir = np.sqrt(hor_direction[0]**2 + hor_direction[1]**2)

    # Cosine of the angle between the two vectors
    cos_theta = dot_dir / (len_hands_dir * len_hor_dir)

    # Limiting to [-1, 1] to avoid errors with np.arcos
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    # Calculating the actual angle in radians
    theta = np.arccos(cos_theta)

    # From radians to degrees
    theta_degrees = np.degrees(theta)

    # Cross product
    cross = hands_direction[0] * hor_direction[1] - hands_direction[1] * hor_direction[0]
    
    if cross > 0:
        return 360 - theta_degrees # hands_dir on the left of hor_dir
    else:
        return theta_degrees # hands_dir on the right or same direction of hor_dir


def get_time(hour_a: float, minute_a: float, second_a: float | None) -> str:
    """Obtains time from the angles of the hands.

    Args:
     hour_a (float): Angle of the hour hand.
     minute_a (float): Angle of the minute hand.
     second_a (float | None): Angle of the second hand.

    Retuns:
     time (str): Formatted string of the extracted time.
    """
    hour = hour_a / 30
    minute = minute_a / 6
    second = second_a / 6

    if (round(hour)*30 - hour_a <= 6) and ((355 < minute_a < 360) or (minute_a < 90)):
        hour = round(hour)
        if hour == 12:
            hour = 0

    if (hour_a - hour*30 <= 6) and (355 < minute_a < 360):
        minute = 0
    
    if (round(minute)*6 - minute_a <= 6) and (second_a < 6):
        minute = round(minute)
        if minute == 60:
            minute = 0

    if (minute_a - minute*30 <= 6) and (354 < second_a < 360):
        second = 0
    
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    time = f"{hour:02d}:{minute:02d}:{second:02d}"

    return time


def detect_time_gray(file_name: str):
    """Tries its best to find the time from the image.

    Args:
        file_name (str): Name of the image.    
    """

    # > Pre-processing of the image
    img, img_gray = open_img(file_name)
    img_gray = cv2.bitwise_not(img_gray)

    # - histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2)
    eq_img = clahe.apply(img_gray)

    # - thresholding
    _, thresholded_img = cv2.threshold(eq_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # - blurring
    blurred_img = cv2.GaussianBlur(thresholded_img, (5, 5), 0)

    # > Searching the clock in the image
    clock_x, clock_y, clock_radius = find_clock(blurred_img)

    # > Searching for the lines
    lines = find_lines(blurred_img)
    # - Grouping the lines
    groups = find_group_lines(lines, clock_x, clock_y, clock_radius)

    # > Finding the hands of the clock
    hands = find_hands(groups, clock_x, clock_y)

    # - Extracting the hour, minute and second hands
    hour, minute, second = sort_hands(hands)

    hour_angle   = get_angle(hour, clock_y)
    minute_angle = get_angle(minute, clock_y)
    second_angle = get_angle(second, clock_y) if second != None else 0.0

    time = get_time(hour_angle, minute_angle, second_angle)

    print(time)


if __name__ == "__main__":
    detect_time_gray("clock1.jpg")
