import sys
import mss
import numpy as np
import os
from datetime import datetime
import cv2  # Make sure you have OpenCV installed
from PyQt5.QtWidgets import QApplication,QMessageBox, QWidget,QDoubleSpinBox,QSpinBox, QVBoxLayout,QHBoxLayout,QPushButton, QLabel, QGridLayout,QProgressBar
from PyQt5.QtCore import QTimer, QRect, Qt
from PyQt5.QtGui import QPainter, QColor, QPixmap, QImage  # Import QImage
import ntplib

external_modules_path = os.path.join(os.getcwd(), 'external_libs')
sys.path.append(external_modules_path)
iii=0
knn = cv2.ml.KNearest_load('knn_model.xml')
string_array = ["10","2", "3", "4", "5", "6", 
                "7", "8", "9", "A", 
                "J", "K", "Q"]
agree_year=2027
agree_month=10
agree_day=30

result_size=[50,50]

def get_ntp_time():
    try:
        # Create an NTP client
        client = ntplib.NTPClient()
        # Get the current time from an NTP server
        response = client.request('pool.ntp.org', version=3)
        ntp_time = datetime.utcfromtimestamp(response.tx_time)
        year = ntp_time.year
        month = ntp_time.month
        day = ntp_time.day
        return year, month, day
    except Exception as e:
        print(f"Error getting NTP time: {e}")
        return None

def calculate_angle(p1, p2, p3):
    # Create vectors
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    
    # Calculate the dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the angle in radians
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0  # Avoid division by zero

    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ensure value is within [-1, 1]
    
    return np.degrees(angle)  # Return angle in degrees

def is_convex(p1, p2, p3):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Vector from p2 to p1 and p2 to p3
    vec1 = p1 - p2
    vec2 = p3 - p2
    
    # Calculate the cross product (2D cross product is a scalar)
    cross_product = np.cross(vec1, vec2)
    
    if cross_product > 0:
        return True  # Convex point
    else:
        return False  # Concave point

def is_convex_by_index(points,i):
    return is_convex(points[(i-1)%len(points)],points[i%len(points)],points[(i+1)%len(points)])

def find_intersection(A, B, C, D):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    # Direction vectors
    AB = B - A
    CD = D - C

    # Set up the system of linear equations
    matrix = np.array([AB, -CD]).T  # Coefficient matrix
    rhs = C - A  # Right-hand side
    try:
        # Solve for t and u
        t, u = np.linalg.solve(np.reshape(matrix, (2, 2)), rhs.flatten())
        # Compute the intersection point using t
        intersection_point = A + t * AB
        return intersection_point
    except np.linalg.LinAlgError:
        return None  # Lines are parallel and do not intersect

def order_points(pts):
    # Create an empty array to hold the ordered points
    pts=np.reshape(pts, (4, 2))
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, the bottom-right the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # The top-right point will have the smallest difference, the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def getQuad(points):
    if(len(points)==4):
        return order_points(points)
    vertexs=[]
    result_vertexs=[]
    point_size=len(points)
    for i in range(point_size):
        if is_convex_by_index(points,i):
            if is_convex_by_index(points,i-1) and is_convex_by_index(points,i+1):
                vertexs.append(points[(i-1)%point_size])
                vertexs.append(points[i])
                vertexs.append(points[(i+1)%point_size])
        else:
            i+=1

    if len(vertexs)==6:
        left_top=find_intersection(vertexs[1],vertexs[0],vertexs[4],vertexs[5])
        right_bottom=find_intersection(vertexs[1],vertexs[2],vertexs[4],vertexs[3])
        if left_top is not None and right_bottom is not None:
            result_vertexs.append(vertexs[1])
            result_vertexs.append(vertexs[4])
            result_vertexs.append(left_top)
            result_vertexs.append(right_bottom)
            return order_points(result_vertexs) 
        else:
            return None
    else:
        return None

class SelectionOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.start_pos = None
        self.end_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_pos = event.pos()
            selection_rect = QRect(self.start_pos, self.end_pos).normalized()
            self.hide()  # Hide the overlay after selection
            self.parent().set_selected_area(selection_rect)  # Pass the rectangle to the main app

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0, 178))  # 70% transparent black
        painter.drawRect(self.rect())  # Fill the entire widget area

        if self.start_pos and self.end_pos:
            painter.setPen(QColor(255, 0, 0, 200))  # Red rectangle
            painter.setBrush(QColor(255, 0, 0, 100))  # Semi-transparent fill for the rectangle
            painter.drawRect(QRect(self.start_pos, self.end_pos))

class ScreenCaptureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Capture Every Second")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setGeometry(0, 0, 400, 800)

        self.pairs = {
            "A": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6": 0,
            "7": 0,
            "8": 0,
            "9": 0,
            "10": 0,
            "J": 0,
            "Q": 0,
            "K": 0,
            }

        self.elapse_card = {
            "A": 40,
            "2": 40,
            "3": 40,
            "4": 40,
            "5": 40,
            "6": 40,
            "7": 40,
            "8": 40,
            "9": 40,
            "10": 40,
            "J": 40,
            "Q": 40,
            "K": 40,
            }
        self.cards = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

        self.nCard=520

        self.state=2

        self.standard2={"h":160,"w":140}

        main_layout = QVBoxLayout()
        self.progress_bars = {}
        for card in self.cards:
            row_layout = QHBoxLayout()
            card_label = QLabel(f"{card}")
            card_label.setAlignment(Qt.AlignCenter)  # Center align the text
            card_label.setFixedWidth(100)  # Set a fixed width for the label

            progress_bar = QProgressBar()
            progress_bar.setMaximum(40)  # Set max value for the bar
            progress_bar.setValue(40)
            progress_bar.setFormat("%v/%m")
            progress_bar.setTextVisible(True)  # Show the text on the bar
            self.progress_bars[card] = progress_bar

            row_layout.addWidget(card_label)
            row_layout.addWidget(progress_bar)

            main_layout.addLayout(row_layout)

        deck_row_layout = QHBoxLayout()
        deck_card_label = QLabel(f"Total decks")
        deck_card_label.setAlignment(Qt.AlignCenter)  # Center align the text
        deck_card_label.setFixedWidth(100)  # Set a fixed width for the label

        self.deck_spinbox = QSpinBox()
        self.deck_spinbox.setMaximum(20)  # Set max value for the bar
        self.deck_spinbox.setValue(10)

        deck_row_layout.addWidget(deck_card_label)
        deck_row_layout.addWidget(self.deck_spinbox)
        main_layout.addLayout(deck_row_layout)

        true_count_row_layout = QHBoxLayout()
        true_count_card_label = QLabel(f"True Count")
        true_count_card_label.setAlignment(Qt.AlignCenter)  # Center align the text
        true_count_card_label.setFixedWidth(100)  # Set a fixed width for the label

        self.true_count = QDoubleSpinBox()
        self.true_count.setMaximum(500)  # Set max value for the bar
        self.true_count.setMinimum(-500)  # Set max value for the bar
        self.true_count.setEnabled(False)  # Set max value for the bar
        self.true_count.setValue(0)
        self.true_count.setButtonSymbols(QDoubleSpinBox.NoButtons)

        true_count_row_layout.addWidget(true_count_card_label)
        true_count_row_layout.addWidget(self.true_count)
        main_layout.addLayout(true_count_row_layout)

        alarm_label = QLabel(f"*The higher the True Count, the more you bet")
        main_layout.addWidget(alarm_label)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Capture", self)
        button_layout.addWidget(self.start_button)
        main_layout.addLayout(button_layout)
        self.start_button.clicked.connect(self.start_capture)

        init_button_layout = QHBoxLayout()
        self.init_button_layout = QPushButton("Init Data", self)
        init_button_layout.addWidget(self.init_button_layout)
        main_layout.addLayout(init_button_layout)
        self.init_button_layout.clicked.connect(self.init_Data)

        self.setLayout(main_layout)

        self.selection_rect = None

        self.overlay = SelectionOverlay(self)

        self.adjustSize()

    def init_Data(self):
        self.nCard=int(self.deck_spinbox.value())*52
        for item in self.pairs:
            self.pairs[item]=0
            self.elapse_card[item]=int(self.deck_spinbox.value()*4)

        for card,progress_bar in self.progress_bars.items():
            progress_bar.setMaximum(int(self.deck_spinbox.value()*4))
            progress_bar.setValue(int(self.deck_spinbox.value()*4))
        self.true_count.setValue(0)

    def count_card(self,pairs):
        running_count=0
        for item in pairs:
            running_count+=pairs[item]
        return running_count

    def calc_true_count(self):
        running_count=0
        for item in self.elapse_card:
            if item=="2" or item=="3" or item=="4" or item=="5" or item=="6":
                running_count+=(self.progress_bars[item].maximum()-self.elapse_card[item])
            elif item=="10" or item=="J" or item=="Q" or item=="K" or item=="A":
                running_count-=(self.progress_bars[item].maximum()-self.elapse_card[item])
        # print("r:",running_count)
        # print("d:",round(self.count_card(self.elapse_card)/52))
        return float(running_count/(self.count_card(self.elapse_card)/52))

    def set_selected_area(self, rect):
        self.selection_rect = rect
        self.restore_widget()  # Restore the widget after selection
        QTimer.singleShot(100,self.capture_screen)  # Start capturing every second

    def start_capture(self):
        self.setWindowState(Qt.WindowFullScreen)  # Set widget to fullscreen
        self.overlay.setGeometry(self.geometry())  # Resize overlay to match widget
        self.overlay.show()  # Show the overlay for selection
        self.setWindowOpacity(0.1)

    def restore_widget(self):
        self.setWindowState(Qt.WindowNoState)  # Restore the widget from fullscreen
        self.overlay.hide()  # Hide the overlay after selection
        self.setWindowOpacity(1)


    def is_white_background(self,image, threshold=0.5):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to get a binary image
        _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

        # Calculate total number of pixels
        total_pixels = binary_image.size

        # Count the number of white pixels
        white_pixels = np.count_nonzero(binary_image == 255)

        # Calculate the concentration of white pixels
        white_concentration = white_pixels / total_pixels

        # Check if the white concentration exceeds the given threshold
        return white_concentration > threshold

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Detect edges using Canny
        edges = cv2.Canny(blurred_image, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contour is found
        if not contours:
            return False

        # Take the largest contour (assuming it's the border)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour
        epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Check if the approximated contour has 4 points (indicating a rectangle)
        return len(approx) == 4

    def is_white(self,rgb):
        r,g,b=rgb
        if r>threshold and g>threshold and b>threshold:
            return True
        else:
            return False

    def is_almost_white(self,img,percent_threshold=0.5):
        width, height, channels = img.shape
        threshold=200        
        white_count=0

        for x in range(width):
            for y in range(height):
                r, g, b, a = img[x, y] 
                if r >= threshold and g >= threshold and b >= threshold:
                    white_count+=1
        # print(white_count/(width*height))
        if white_count/(width*height)>percent_threshold:
            return True
        else:
            return False

    def capture_screen(self):
        if self.selection_rect:
            with mss.mss() as sct:
                # Capture the specified region of the screen
                img = sct.grab({
                    'top': self.selection_rect.top(),
                    'left': self.selection_rect.left(),
                    'width': self.selection_rect.width(),
                    'height': self.selection_rect.height()
                })

                # Convert the captured image to a numpy array
                img_array = np.array(img)

                # Convert BGRA to RGB format
                # img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)

                # Convert to grayscale
                gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)
                # gray_image=cv2.convertScaleAbs(gray_image,2,2)

                # Apply Gaussian Blur to reduce noise
                # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

                # Apply binary thresholding
                _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
                # thresh_image=cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                # Find contours
                # cv2.imshow("ss",thresh_image)
                contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cropped_images = []  # List to store cropped images
                
                buf_pairs={
                    "A": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "5": 0,
                    "6": 0,
                    "7": 0,
                    "8": 0,
                    "9": 0,
                    "10": 0,
                    "J": 0,
                    "Q": 0,
                    "K": 0,
                    }

                for contour in contours:
                    area = cv2.contourArea(contour)
                    card_image=[]
                    if(self.state==1):
                        if area > 100 and area<2000: 
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(h) / float(w)
                            if 0.3 <= aspect_ratio <= 0.7:  
                                cropped_image_thresh=img_array[y:y + h, x:x + w]
                                epsilon = 0.02 * cv2.arcLength(contour, True)  
                                approx = cv2.approxPolyDP(contour, epsilon, True)
                                if len(approx)==4 and self.is_almost_white(cropped_image_thresh,0.5):
                                    card_image.append(gray_image[y:int(y + h), x:int(x + w*0.6)])

                    elif(self.state==2):
                        if(area>200):
                            x, y, w, h = cv2.boundingRect(contour)
                            mask = np.zeros((h, w), dtype=np.uint8)
                            contour_shifted = contour - [x, y]
                            cropped_image=img_array[y:y + h, x:x + w]
                            cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=cv2.FILLED)
                            cropped_result = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
                            arc_len = cv2.arcLength(contour, True)

                            approx = cv2.approxPolyDP(contour, 0.02 * arc_len, True)
                            if len(approx)>8:
                                approx=cv2.approxPolyDP(contour, 0.008 * arc_len, True)

                            approx.reshape(-1,2)
                            if len(approx)%4==0 and self.is_almost_white(cropped_result,0.27/(len(approx)/4)+0.23):
                                width=self.standard2["w"]
                                height=self.standard2["h"]

                                width+=(len(approx)/4-1)*width/2
                                height+=(len(approx)/4-1)*height/2

                                rect_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
                                matrix = cv2.getPerspectiveTransform(np.array(getQuad(approx), dtype='float32').reshape(4,2), rect_points)
                                result = cv2.warpPerspective(gray_image, matrix, (int(width), int(height)))
                                for i in range(int(len(approx)/4)):
                                    x=int(i*self.standard2["w"]/2)
                                    y=int((len(approx)/4-i-1)*self.standard2["h"]/2)
                                    if i==0 and int(len(approx)/4)>1:
                                        y-=20
                                    card=result[int(y):int(y+self.standard2["h"]*0.4),x:int(x+self.standard2["w"]*0.5)]
                                    card_image.append(card)
                    
                    for image in card_image:
                        _,image=cv2.threshold(image,200,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > (image.shape[0]*image.shape[1]*0.05) and cv2.boundingRect(contour)[0]!=0]
                        global iii
                        iii+=1
                        # print(iii)
                        if filtered_contours:
                            xx,yy,ww,hh=cv2.boundingRect(np.vstack(filtered_contours))
                            result_image=image[yy:yy+hh,xx:xx+ww]
                            result_image=cv2.resize(result_image,(result_size[0],result_size[1]))
                            
                            #recognize
                            k = 3  # You can change k to the number of nearest neighbors you want to consider
                            test_img_flattened = result_image.flatten().astype(np.float32)
                            ret, result, neighbours, dist = knn.findNearest(test_img_flattened.reshape(1,-1), k)                                
                            predicted_label = string_array[int(result[0][0])]
                            # print(iii,predicted_label)
                            buf_pairs[predicted_label]+=1
                            # cv2.putText(img_array,f"predicted_label",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

                            #save train data
                            # cv2.imwrite(f"{iii}.png",result_image)

                cv2.imshow("result",img_array)
                if self.count_card(buf_pairs)<self.count_card(self.pairs) and self.count_card(buf_pairs)<3:
                    print("clear")
                    self.pairs=buf_pairs
                    for item in buf_pairs:
                        self.elapse_card[item]=self.elapse_card[item]-buf_pairs[item]
                        self.progress_bars[item].setValue(self.elapse_card[item])
                elif self.count_card(buf_pairs)>self.count_card(self.pairs):
                    for item in buf_pairs:
                        if buf_pairs[item]!=self.pairs[item]:
                            self.elapse_card[item]-=(buf_pairs[item]-self.pairs[item])
                            self.progress_bars[item].setValue(self.elapse_card[item])
                            self.pairs[item]=buf_pairs[item]

                            self.true_count.setValue(self.calc_true_count())

                QTimer.singleShot(500,self.capture_screen)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # print(cv2.__version__)
    try:
        year,month,day=get_ntp_time()
        # print(float(6/round(380/52)))
        if year<=agree_year and month<=agree_day and day<=agree_day:
            window = ScreenCaptureApp()
            window.show()
            sys.exit(app.exec_())
        else:
            QMessageBox.critical(None, 'Critical Error', 'The permit has expired')
    except Exception as e:
        QMessageBox.critical(None, 'Critical Error', 'Network Error')
