import tkinter as tk
from tkinter import filedialog, Text, RIGHT, Y, messagebox
from collections import deque
import pandas as pd
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import argparse

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        self.video_frame = tk.Label(root)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(self.video_frame, width=600, height=400)
        self.canvas.pack()

        self.text_frame = tk.Frame(root)
        self.text_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ns")

        self.text_area = Text(self.text_frame, wrap='word', state='disabled', width=40, height=25)
        self.text_area.pack(side=tk.LEFT, fill=tk.Y)

        self.scrollbar = tk.Scrollbar(self.text_frame, command=self.text_area.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.text_area['yscrollcommand'] = self.scrollbar.set

        self.file_button = tk.Button(root, text="Choose Video", command=self.open_video)
        self.file_button.grid(row=1, column=0, pady=10)

        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video",
            help="path to the (optional) video file")
        ap.add_argument("-b", "--buffer", type=int, default=24,
            help="max buffer size")
        self.args = vars(ap.parse_args())

        self.trackers = (deque(maxlen=self.args["buffer"]),
                         deque(maxlen=self.args["buffer"]),
                         deque(maxlen=self.args["buffer"]),
                         deque(maxlen=self.args["buffer"]))

        self.colors = {
            'yellow': 0,
            'pink': 0,
            'green': 0,
            'white': 0
        }

        self.video_source = ""
        self.delay = 5
        self.stop = False
        self.error_window_size = 7
        self.entered_just_now = False
        self.entered = 0
        self.df = None

    def get_color(self, point, frame):
        self.hsv_ranges = {
            'yellow': ([0, 94, 43], [50, 255, 203]),
            'pink': ([0, 90, 163], [31, 255, 255]),
            'green': ([50, 82, 9], [116, 255, 144]),
            'white': ([0, 0, 70], [65, 105, 255])
        }

        # self.hsv_ranges = {
        #     'yellow': ([20, 100, 100], [30, 255, 255]),
        #     'pink': ([140, 100, 100], [170, 255, 255]),
        #     'green': ([35, 100, 100], [85, 255, 255]),
        #     'white': ([0, 0, 200], [180, 20, 255])
        # }

        x, y = point
        rgb = frame[y, x]
        bgr = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2BGR)[0][0]
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        color = "Invalid"
        for name, (lower, upper) in self.hsv_ranges.items():
            if all(lower[i] <= hsv[i] <= upper[i] for i in range(3)):
                color = name
                break

        return (color, bgr, hsv)

    def track_balls(self, frame):
        results = self.model(frame)[0]
        balls_detected = 0
        xyxy = []
        for result in results.boxes:
            if result.cls == 0:  # 0 is 'Ball'
                confidence = result.conf[0]
                if confidence > 0.5:
                    balls_detected += 1
                    xyxy.append(map(int, result.xyxy[0]))

        return (balls_detected, xyxy)

    def draw_trail(self, frame):
        for j, tracker in enumerate(self.trackers):
            for i in range(1, len(tracker)):
                
                # Checking exit condition
                if tracker[i-1] is None and tracker[i] is None:
                    if i > self.error_window_size:
                        error_window = list(set([tracker[i-j] for j in range(1, self.error_window_size)]))
                        if len(error_window) == 1 and error_window[0] is None and (j+1) in list(self.colors.values()) and not self.entered_just_now:
                            # print(self.frame_number, "---------", error_window)
                            color = list(self.colors.keys())[list(self.colors.values()).index(j+1)]
                            self.print_text(f"Frame: {self.frame_number}\nExit of {color}")

                            # self.df.append({'Time (sec)': int(self.frame_number/self.fps), 
                            #                 'Color': color, 
                            #                 'Event': 'Exit', 
                            #                 'Quadrant': j+1}, 
                            #                 ignore_index=True)
                            
                            self.df = pd.concat([pd.DataFrame([[int(self.frame_number/self.fps),
                                                                color,
                                                                "Exit",
                                                                j+1]], 
                                                                columns=self.df.columns), self.df], ignore_index=True)
                            self.colors[color] = 0
                    continue

                if tracker[i-1] is None or tracker[i] is None:
                    continue
                
                # otherwise draw the connecting lines
                frame = cv2.line(frame,
                                (int(tracker[i-1][0]), int(tracker[i-1][1])),
                                (int(tracker[i][0]), int(tracker[i][1])),
                                (0, 0, 255), 2)
        
        return frame

    def get_status(self, quadrant):
        if len(self.trackers[quadrant]) <= self.error_window_size:
            return "entry"
        else:
            error_window = list(set([self.trackers[quadrant][-i] for i in range(1, self.error_window_size)]))
            if (len(error_window) == 1 and error_window[0] is None):
                return "entry"
            else:
                return "prevail"

    def info(self, point):
        x, y = point
        status = ""
        if x >= 245 and x <= 390 and y >= 5 and y <= 195:
            status = self.get_status(2)
            self.trackers[2].appendleft(point)
            return 2, status
        elif x >= 245 and x <= 390 and y >= 195 and y <= 375:
            status = self.get_status(1)
            self.trackers[1].appendleft(point)
            return 1, status
        elif x >= 390 and x <= 550 and y >= 5 and y <= 195:
            status = self.get_status(3)
            self.trackers[3].appendleft(point)
            return 3, status
        elif x >= 390 and x <= 550 and y >= 195 and y <= 375:
            status = self.get_status(0)
            self.trackers[0].appendleft(point)
            return 0, status
        else:
            return -1, 'outside'

    def open_video(self):
        self.video_source = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not self.video_source:
            return
        
        self.vid = cv2.VideoCapture(self.video_source)
        self.writer = cv2.VideoWriter("saved\\processed_vid.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (600, 400))
        #self.vid.set(cv2.CAP_PROP_POS_FRAMES, 1500)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.delay = 5
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Unable to open the video file.")
        else:
            self.stop = False
            self.df = pd.DataFrame(columns=['Time (sec)', 'Color', 'Event', 'Quadrant'])
            self.print_text("Video opened successfully.\nSaving logs to 'log.csv'")
            self.play_video()

    def play_video(self):
        if not self.vid or not self.vid.isOpened():
            messagebox.showwarning("Warning", "No video has been opened.")
            return
        
        self.stop = False
        self.model = YOLO("runs\\detect\\train\\weights\\best.pt")
        self.print_text("Model loaded successfully.")
        self.print_text("Playing video: " + self.video_source)
        self.frame_number = 0
        self.update_frame()

    def stop_video(self):
        self.stop = True

    def update_frame(self):
        if self.vid and not self.stop:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (600, 400))

                # self.print_text("Frame: " + str(self.frame_number))
                # self.frame_number += 1

                # if self.frame_number == 5:
                #     squares = self.detect_squares(frame)
                #     print(squares)
                #     # cv2.drawContours(frame, squares, -1, (0, 255, 0), 2)

                # Marking the quadrants
                cv2.rectangle(frame, (245, 5), (390, 195), (0, 255, 0), 2)
                cv2.putText(frame, "Quadrant 3", (245, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.rectangle(frame, (245, 195), (390, 375), (0, 0, 255), 2)
                cv2.putText(frame, "Quadrant 2", (245, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.rectangle(frame, (390, 5), (550, 195), (255, 0, 0), 2)
                cv2.putText(frame, "Quadrant 4", (390, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.rectangle(frame, (390, 195), (550, 375), (255, 255, 0), 2)
                cv2.putText(frame, "Quadrant 1", (390, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                balls_info = self.track_balls(frame)

                if balls_info[0] == 0:
                    for i in range(4):
                        self.trackers[i].appendleft(None)

                # bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                # final_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

                # # Loop through each range and create masks
                # for (lower, upper) in self.hsv_ranges:
                #     lower_bound = np.array(lower, dtype=np.uint8)
                #     upper_bound = np.array(upper, dtype=np.uint8)
                #     # Create the mask for the current range
                #     mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
                #     # Combine the mask with the final mask using bitwise OR
                #     final_mask = cv2.bitwise_or(final_mask, mask)

                # # Apply the final mask to the image
                # result = cv2.bitwise_and(bgr_frame, bgr_frame, mask=final_mask)
                # frame = result

                point_quads = []

                for i in range(balls_info[0]):
                    x1, y1, x2, y2 = balls_info[1][i]
                    # self.print_text(f"Ball at ({x1}, {y1}) to ({x2}, {y2})")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    center = (int((x1+x2)/2), int((y1+y2)/2))
                    pixel_color = frame[center[1], center[0]]
                    hsv_color = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_RGB2HSV)
                    # if balls_info[0] == 4:
                    #     print("--------------RGB ", pixel_color)
                    #     print("--------------HSV ", hsv_color)
                    quad_status = self.info(center)
                    point_quads.append(quad_status[0])

                    #print(f"Status at {quad_status}: {i} i and frame {self.frame_number}")

                    # if quad_status[1] == 'entry':
                    #     self.print_text("Attempting entry: \n\t" + str(self.get_color(center, frame)[2][0]) + ", " + str(self.get_color(center, frame)[2][1]) + ", " + str(self.get_color(center, frame)[2][2]))
                    #     if not self.entered_just_now:
                    #         print(self.get_color(center, frame)[2])
                    #         if self.colors[self.get_color(center, frame)[0]] == 0:
                    #             self.print_text(f"Frame: {self.frame_number}\nEntry of {self.get_color(center, frame)[0]} in Quadrant {quad_status[0] + 1}")
                    #             self.colors[self.get_color(center, frame)[0]] = quad_status[0] + 1
                    #             self.entered_just_now = True

                    if quad_status[1] == 'entry' and not self.entered_just_now:
                        self.print_text(str(self.get_color(center, frame)[2][0]) + ", " + str(self.get_color(center, frame)[2][1]) + ", " + str(self.get_color(center, frame)[2][2]))
                        print(self.get_color(center, frame)[2])
                        if self.colors[self.get_color(center, frame)[0]] == 0:
                            color = self.get_color(center, frame)[0]

                            self.print_text(f"Frame: {self.frame_number}\nEntry of {color} in Quadrant {quad_status[0] + 1}")
                            # self.df.append({'Time (sec)': int(self.frame_number/self.fps), 
                            #                 'Color': color, 
                            #                 'Event': 'Entry', 
                            #                 'Quadrant': quad_status[0]+1}, 
                            #                 ignore_index=True)
                            
                            self.df = pd.concat([pd.DataFrame([[int(self.frame_number/self.fps),
                                                                color,
                                                                "Entry",
                                                                quad_status[0]+1]], 
                                                                columns=self.df.columns), self.df], ignore_index=True)
                                                        
                            self.colors[self.get_color(center, frame)[0]] = quad_status[0] + 1
                            self.entered_just_now = True

                if self.entered_just_now:
                    self.entered += 1
                    if self.entered > 105:
                        self.entered_just_now = False
                        self.entered = 0

                no_point_quads = list(set(range(4)) - set(point_quads))
                
                # Noning the trackers of quadrats with no balls
                for i in range(len(no_point_quads)):
                    self.trackers[no_point_quads[i]].appendleft(None)

                frame = self.draw_trail(frame)

                # self.print_text("Frame: " + str(self.frame_number))
                # self.print_text("Trackers: " ", " + 
                #                 str(self.trackers[0][-1]) + ", " + 
                #                 str(self.trackers[1][-1]) + ", " + 
                #                 str(self.trackers[2][-1]) + ", " + 
                #                 str(self.trackers[3][-1]))
                self.frame_number += 1

                image = Image.fromarray(frame)
                self.writer.write(frame)
                image_tk = ImageTk.PhotoImage(image=image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
                self.canvas.image_tk = image_tk
                self.root.after(self.delay, self.update_frame)
            else:
                self.df.to_csv("saved\\log.csv", index=False)
                self.vid.release()
                self.writer.release()
                self.canvas.image_tk = None
                self.print_text("Video has ended.")
                self.print_text("Log file saved as 'log.csv' in saved folder.")

    def print_text(self, text):
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.config(state='disabled')
        self.text_area.see(tk.END)

    def terminate(self):
        if self.df is not None:
            self.df.to_csv("saved\\log.csv", index=False)
        root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", player.terminate)
    root.mainloop()
