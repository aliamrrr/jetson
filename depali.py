
########################################################### CLASSES ########################################################################################################

class Object_Detected:
    '''
       Class about every objects that were detected with yolo. With an id, position , previous position, class , confidence .. .  
    '''
  
    id = 0
    position = (0,0)
    conf = 0.0
    cls = 0
    position_defference_between_frame = 0
    time_detection_previous_frame = 0.0
    time_detection_frame = 0.0
    speed = []
    bounding_box_length = 0
    previous_position = (0,0)
    time_entry_zone = 0.0
    time_out_of_zone = 0.0
    time_diffence_zone = 0.0
    def __init__(self,_id,pt,_conf,_cls,bounding_box_diff):
        self.id = _id
        self.position = pt
        self.conf = _conf
        self.cls = _cls
        self.bounding_box_length = bounding_box_diff
        self.speed = []

    def update_previous_position(self):
         '''
            update the position. Actual position is now previous position
         '''
         self.pos_deque.append(self.position)
    def update_previous_time_of_detection(self):
         '''
            Update the timezone where the object is detected.  
         '''
         self.time_detection_previous_frame = self.time_detection_frame
    def estimateSpeed(self): 
        '''
            Function that calculates speed of each vehicles detected
        '''
        import math

        # Calculate the distance between previous position and actual one
        height = self.position[0] - self.previous_position[0]
        width = self.position[1] - self.previous_position[1]        
        distance_in_pixels = math.sqrt(math.pow(height,2) + math.pow(width,2))
        if self.cls == 2 : 
            pixels_per_meter = self.bounding_box_length/4.7
        if self.cls == 1 : 
            pixels_per_meter = self.bounding_box_length
        if self.cls == 3:
            pixels_per_meter = self.bounding_box_length/2.2
        if self.cls == 4:
            pixels_per_meter = self.bounding_box_length/14
        if self.cls == 7:
            pixels_per_meter = self.bounding_box_length/9
        else : 
            pixels_per_meter = 15
        distance_in_meters = distance_in_pixels/pixels_per_meter

        #Calculate the time between previous position and previous one
        Time_  = self.time_detection_frame - self.time_detection_previous_frame 

        speed_mps = distance_in_meters/Time_

        #Estimation speed in km/h 
        speed_kmph = speed_mps*(3600/1000)
        if speed_kmph<100 : 
            self.speed.append(speed_kmph)
    def difference_time_through_zone(self):
           self.time_diffence_zone = self.time_out_of_zone - self.time_entry_zone




class traffic_camera : 
    ''' 
        Define the ITS camera object with its parameters like angle, position and direction
    '''
    angle_from_zone = 0
    direction_from_zone = '' #Direction of the vehicle from the camera
    camera_vector = (0,0)
    def __init__(self,_id_,__name__,__loc__):
          self.id = _id_
          self.name = __name__
          self.coords = __loc__
          #self.angle_from_zone = angle


class detection_zone : 
    '''
        Define a detection zone with lat, long points from its
    '''

    def __init__(self,lat1,long1,lat2,long2,lat3,long3,lat4,long4):
          self.latitude1= lat1
          self.longitude1 = long1
          self.latitude2 = lat2
          self.longitude2 = long2
          self.latitude3 = lat3
          self.longitude3 = long3
          self.latitude4 = lat4
          self.longitude4 = long4
    def barycentre(self):
        '''
            Get barycentre of the detection zone
        '''
        polygone = [(self.latitude1,self.longitude1),(self.latitude2,self.longitude2),(self.latitude3,self.longitude3),(self.latitude4,self.longitude4)]
        # Initialiser les sommes pour x et y
        somme_x = 0
        somme_y = 0
        
        # Calculer les sommes
        for point in polygone:
            somme_x += point[0]
            somme_y += point[1]
        
        # Calculer les moyennes
        moyenne_x = somme_x / len(polygone)
        moyenne_y = somme_y / len(polygone)   
        return moyenne_x,moyenne_y


class vehicle_counter : 
    '''
        Define a counter
    '''

    def __init__(self):
        self.pedestrian_counter = []
        self.car_counter = []
        self.truck_counter = []
        self.bicycle_counter = []
        self.moto_counter = []
        self.bus_counter  = []
    def add_car(self,car):
        self.car_counter.append(car)
    def add_truck(self,truck):
        self.truck_counter.append(truck)
    def add_bicycle(self,bicycle):
        self.bicycle_counter.append(bicycle)
    def add_moto(self,moto):
        self.moto_counter.append(moto)
    def add_bus(self,bus):
        self.bus_counter.append(bus)
    def add_pedestrian(self,pedestrian):
        self.pedestrian_counter.append(pedestrian)

    def remove_car(self,car):
        self.car_counter.remove(car)
    def remove_truck(self,truck):
        self.truck_counter.remove(truck)
    def remove_bicycle(self,bicycle):
        self.bicycle_counter.remove(bicycle)
    def remove_moto(self,moto):
        self.moto_counter.remove(moto)
    def remove_bus(self,bus):
        self.bus_counter.remove(bus)
    def remove_pedestrian(self,pedestrian):
        self.pedestrian_counter.remove(pedestrian)



class vehicle_counter : 
    '''
        Define a counter
    '''

    def __init__(self):
        self.pedestrian_counter = []
        self.car_counter = []
        self.truck_counter = []
        self.bicycle_counter = []
        self.moto_counter = []
        self.bus_counter  = []
    def add_car(self,car):
        self.car_counter.append(car)
    def add_truck(self,truck):
        self.truck_counter.append(truck)
    def add_bicycle(self,bicycle):
        self.bicycle_counter.append(bicycle)
    def add_moto(self,moto):
        self.moto_counter.append(moto)
    def add_bus(self,bus):
        self.bus_counter.append(bus)
    def add_pedestrian(self,pedestrian):
        self.pedestrian_counter.append(pedestrian)

    def remove_car(self,car):
        self.car_counter.remove(car)
    def remove_truck(self,truck):
        self.truck_counter.remove(truck)
    def remove_bicycle(self,bicycle):
        self.bicycle_counter.remove(bicycle)
    def remove_moto(self,moto):
        self.moto_counter.remove(moto)
    def remove_bus(self,bus):
        self.bus_counter.remove(bus)
    def remove_pedestrian(self,pedestrian):
        self.pedestrian_counter.remove(pedestrian)


class Agregation :
    '''
        Define the agregation object with parameters that were sent from its
    '''

    property = '' # Property of the condition that was defined in its about (type of vehicles,..)
    condition = '' # if we want to count cars / trucks / pedestrians ....
    bearing_vector = (0,0) # Vector the goes from the the barycentre to the direction of the bearing parameters
    def __init__(self,id,start_date,end_date,_freq,_period,_bearing,_bearing_tolerance,lat1,long1,lat2,long2,lat3,long3,lat4,long4,camera_id,camera_name,camera_loc,prop,cond):
          self.id = id # id de l'agregation
          self.camera = traffic_camera(camera_id,camera_name,camera_loc) # Define the camera
          self.agregation_start_date = datetime.strptime(start_date, "%Y-%m-%d") # Date de debut
          self.agregation_start_date = self.agregation_start_date.date() # Date de sortie
          self.agregation_end_date =  datetime.strptime(end_date, "%Y-%m-%d")
          self.agregation_end_date = self.agregation_end_date.date()
          self.frequency = _freq # Frequence en minutes
          self.period = _period # Periode en minutes
          self.end_period = self.period * 60 # Calcul de la fin de la periode selon le temps de la video
          self.bearing = _bearing # cap de direction
          self.bearning_tolerance = _bearing_tolerance
          self.detection_zone = detection_zone(lat1,long1,lat2,long2,lat3,long3,lat4,long4) # zone de detection
          self.counter = vehicle_counter() #Definir des compteurs de chaque agregation
          
          if prop != '': # If the property is not empty and was actually filled
                self.property = prop
                self.condition = cond

          
    def get_angle_camera_from_zone(self):
            '''
                Get the angle from the barycentre to the camera. The goal is to calculate a vector that goes from the barycentre
            '''
            #Transform points to longitude and latitude
            point2 = getattr(self.camera,'coords')     
            point1 = self.detection_zone.barycentre()
            lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
            lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
            
            # Calculer la différence de longitude
            dlon = lon2 - lon1
            
            # Calculer l'angle
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            angle = math.atan2(y, x)
            
            # Convertir l'angle en degrés
            angle = math.degrees(angle)
            
            if angle < 0:
                angle += 360
            
            setattr(self.camera,'angle_from_zone',angle)  




    def compare_vectors(self):
          
          '''
                In this function, the goal is to compare the camera vector and the bearing vector.

                We get the scalar product. If the scalar product is + , it means these vectors have the same direction. Thus, from the camera point of view, it goes south

                If the scalar product is - , it means the vectors do not have the same direction. From camera point of view, it goes north
          '''
          import numpy as np
          bearing_points_list = []
          scalar_product = []
          # Use functions to calculate the important parameters that will be used in comparaison 
          self.get_angle_camera_from_zone()
          xca,xcb = get_points_vector(getattr(self.camera,'angle_from_zone'),self.detection_zone)
          bearing_points_list.append(get_points_vector(self.bearing ,self.detection_zone))
          bearing_points_list.append(get_points_vector(self.bearing+self.bearning_tolerance ,self.detection_zone))
          bearing_points_list.append(get_points_vector(self.bearing-self.bearning_tolerance ,self.detection_zone))
          x_bary,y_bary = self.detection_zone.barycentre() # baryucentre point vector

          # Use function that will transform points taht were calculated into functions
          for bearing in bearing_points_list : 
            x_B,y_B = bearing
               
            start_point = convert_to_cartesian(x_bary,y_bary)
            end_point_camera = convert_to_cartesian(xca,xcb)
            end_point_bearing = convert_to_cartesian(x_B,y_B)
            vector_BC = calculate_vector(start_point, end_point_camera)
            vector_BB = calculate_vector(start_point, end_point_bearing)

            # Get scalar product
            scalar_product.append(np.dot(vector_BC , vector_BB))
          #produit_scalaire = sum(x * y for x, y in zip(getattr(self.camera,'camera_vector'), self.bearing_vector))
          
            all_positive = all(num > 0 for num in scalar_product)
            all_negative = all(num < 0 for num in scalar_product)

            if all_positive:
                setattr(self.camera,'direction_from_zone','S') # S for south
                print('sud')
                print("Les vecteurs pointent dans la même direction générale.")
                
            elif all_negative:
                setattr(self.camera,'direction_from_zone','N') # N for north
                print("NORD.")
            # else:
            #     setattr(self.camera,'direction_from_zone','B') # B for both
            #     print("BOTH.")
        #   # if scalar product is positive, direction goes south
        #   else:
        #         setattr(self.camera,'direction_from_zone','P')
        #         x1,y1 = calculer_milieu((getattr(self.detection_zone,'lat1'),getattr(self.detection_zone,'long1')),(getattr(self.detection_zone,'lat2'),getattr(self.detection_zone,'lat2')))
        #         #x2,y2 = calculer_milieu((getattr(self.detection_zone,'lat3'),getattr(self.detection_zone,'long3')),(getattr(self.detection_zone,'lat4'),getattr(self.detection_zone,'lat4')))
        #         point1 = (x1,y1)
        #         point2 =  self.detection_zone.barycentre()
        #         lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        #         lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
                
        #         # Calculer la différence de longitude
        #         dlon = lon2 - lon1
                
        #         # Calculer l'angle
        #         y = math.sin(dlon) * math.cos(lat2)
        #         x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        #         angle = math.atan2(y, x)
                
        #         # Convertir l'angle en degrés
        #         angle = math.degrees(angle)
                
        #         # Normaliser l'angle pour qu'il soit entre 0 et 360
        #         angle = (angle + 360) % 360
        #         bearing = math.radians(angle)
        #         x_A,y_A  = point1      
        #         x_B = x_A + 2 * math.cos(bearing)
        #         y_B = y_A + 2 * math.sin(bearing) 
        #         scalar = np.dot(getattr((x_B,y_B), self.bearing_vector))
        #         if scalar > 0:
        #             setattr(self.camera,'direction_from_zone','W')
        #             #print("Les vecteurs pointent dans la même direction générale.")
        #         elif scalar < 0:
        #             setattr(self.camera,'direction_from_zone','E')
        #             #print("Les vecteurs pointent dans des directions opposées.")

######################################################################  FONCTIONS  ############################################################################################################################

import math
from datetime import datetime,date
import cv2

def get_token():
    """
        Get token from ITS
    """
    from authlib.integrations.requests_client import OAuth2Session
    token_endpoint = 'https://sso.labatosbordeaux.eu/auth/realms/cdp/protocol/openid-connect/token'
    session = OAuth2Session('its', '8ab231c5-1335-4739-a8a5-28415a9cf347')
    token = session.fetch_token(token_endpoint, username='ali-anass.amradouch@eviden.com', password='142003Al@',grant_type='password')
    access_token = token['access_token']
    return access_token

def get_agregation_its():
    """
        Get agregations from ITS
    """
    import requests
    agregation_list = []
    access_token = get_token()
    response = requests.get(
    'https://its.labatosbordeaux.eu/back/cdpits/aggregationRequest', headers={'Authorization': f'Bearer {access_token}'})
    agregation = response.json()
    for agg in agregation :
        if agg['type'] == 'cameraType':
                agregation_list.append(agg)
    return agregation_list

def get_camera_agregation(agregation_list):
        """
                get camera agregation from ITS 

                agregation_list : list of all agregations from ITS
        """       
        
        import requests
        access_token = get_token()
        id_cam = agregation_list[-1]['camera_id']
        camera_list = requests.get(
        f'https://its.labatosbordeaux.eu/back/cdpits/trafficCamera/{id_cam}', headers={'Authorization': f'Bearer {access_token}'}).json()
        return camera_list


def add_video_counter(_obj,agreg):
      """
            Add objects to the counter lists 

            _obj : Object in the object_list

            agreg :  The agregation      
      
      """
      if calculate_direction(getattr(_obj,'position'),getattr(_obj,'previous_position')) == 'South':                                          
                                                    # If the object is car / truck / pedestrian ..... and has not been counted before, we add this object to a list in order to count it
                                                        if getattr(_obj,'cls')==0:
                                                                pedestrian_counter = getattr(getattr(agreg,'counter'),'pedestrian_counter')
                                                                if getattr(_obj,'id') not in pedestrian_counter:
                                                                    if getattr(_obj,'conf')>=0.5:
                                                                        getattr(agreg,'counter').add_pedestrian(getattr(_obj,'id'))
                                                        if getattr(_obj,'cls')==1:
                                                                bicycle_counter = getattr(getattr(agreg,'counter'),'bicycle_counter')
                                                                if getattr(_obj,'id') not in bicycle_counter:
                                                                    if getattr(_obj,'conf')>=0.5:
                                                                       getattr(agreg,'counter').add_bicycle(getattr(_obj,'id'))    
                                                        if getattr(_obj,'cls')==2:
                                                                car_counter = getattr(getattr(agreg,'counter'),'car_counter')
                                                                if getattr(_obj,'id') not in car_counter:
                                                                    if getattr(_obj,'conf')>=0.5:
                                                                        getattr(agreg,'counter').add_car(getattr(_obj,'id'))     
                                                        if getattr(_obj,'cls')==3:
                                                                moto_counter = getattr(getattr(agreg,'counter'),'moto_counter')
                                                                if getattr(_obj,'id') not in moto_counter:
                                                                    if getattr(_obj,'conf')>=0.5:
                                                                        getattr(agreg,'counter').add_moto(getattr(_obj,'id'))
                                                        if getattr(_obj,'cls')==4:
                                                                bus_counter = getattr(getattr(agreg,'counter'),'bus_counter')
                                                                if getattr(_obj,'id') not in bus_counter:
                                                                    if getattr(_obj,'conf')>=0.5:
                                                                        getattr(agreg,'counter').add_bus(getattr(_obj,'id'))   
                                                        if getattr(_obj,'cls')==7:
                                                                truck_counter = getattr(getattr(agreg,'counter'),'truck_counter')
                                                                if getattr(_obj,'id') not in truck_counter:
                                                                    if getattr(_obj,'conf')>=0.5:
                                                                        getattr(agreg,'counter').add_truck(getattr(_obj,'id')) 




def Average(lst): 
    return sum(lst) / len(lst) 

def get_points_vector(angle,zone):
        """
            Get the camera vector from angle between barycentre and the camera position
        """
        # Conversion en radians

        bearing_camera = math.radians(angle)

        xc_A,yc_A  = zone.barycentre()        

        # Calcul des coordonnées du point B
        xc_B = xc_A + 2 * math.cos(bearing_camera)
        yc_B = yc_A + 2 * math.sin(bearing_camera)
        return  (xc_B,yc_B)

def convert_to_cartesian(latitude, longitude):
    # Convertir latitude et longitude en radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Rayon de la Terre en mètres
    R = 6371000

    # Convertir les coordonnées en cartésien
    x = R * math.cos(lat_rad) * math.cos(lon_rad)
    y = R * math.cos(lat_rad) * math.sin(lon_rad)
    z = R * math.sin(lat_rad)

    return x, y, z                
def calculate_vector(start_point, end_point):
    # Calculer les composantes du vecteur
    vector = [end_point[i] - start_point[i] for i in range(3)]
    return vector

def calculer_milieu(point1, point2):
    # Calculer la moyenne des latitudes et des longitudes
    latitude_moyenne = (point1[0] + point2[0]) / 2
    longitude_moyenne = (point1[1] + point2[1]) / 2
    
    return latitude_moyenne, longitude_moyenne
def  calculate_direction(current_coords, past_coords):
    # current_coords and past_coords are tuples (x, y)
    import numpy as np

    # Calculate the direction based on the change in coordinates
    direction = np.subtract(current_coords, past_coords)
    cardinal_direction = get_cardinal_direction(direction)

    return cardinal_direction

def get_cardinal_direction(direction):
    x, y = direction

    if abs(x) > abs(y):
        if x > 0:
            return "East"
        else:
            return "West"
    else:
        if y > 0:
            return "South"
        else:
            return "North"

def intersect(A,B,C,D): 
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def draw_lines(img,p1,p2,color):
    cv2.line(img,p1,p2,color,3)

import cv2
import numpy as np

def process_frame(frame, prev_rect):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Appliquer la détection de contours de Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Appliquer la transformation de ligne de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

    # Dessiner le rectangle suivant la direction de la route
    rect_t,rect_b = draw_following_rectangle(frame, lines, prev_rect)

    return frame, rect_t,rect_b

import importlib
import cv2
import numpy as np

def zone_drawing(video_path):
    
    rect_top_list = []
    rect_bottom_list = []
    Break = False
    frame_counter = 0
    # Ouvrir la capture vidéoprev_rect
    cap = cv2.VideoCapture(video_path)  # Remplacez par votre fichier vidéo
    #cap  = load_video(video_path)

    # Initialiser les coordonnées du rectangle précédent
    top_rect = None
    bottom_rect = None
    prev_rect = None


    while cap.isOpened():
        frame_counter +=1
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480)) 
        if not ret:
            break
        # Traiter le frame
        frame_traite, top_rect,bottom_rect = process_frame(frame, prev_rect)
        for top_points in top_rect : 
            rect_top_list.append(top_points)
        for bottom_points in bottom_rect : 
            rect_bottom_list.append(bottom_points)
        if frame_counter > 600:
            # Convert the list of tuples to a numpy array
            array_top = np.array(rect_top_list)
            

            # Calculate the mean along axis 0 (column-wise)
            mean_top= np.mean(array_top, axis=0)
            
            array_bottom = np.array(rect_bottom_list)

            # Calculate the mean along axis 0 (column-wise)
            mean_bottom= np.mean(array_bottom, axis=0)
            Break = True
            
            #cv2.rectangle(frame, tuple(np.round(mean_top).astype(int)), tuple(np.round(mean_bottom).astype(int)), (0, 85, 200), 5)4
            points  = tuple(np.round(mean_top).astype(int)), tuple(np.round(mean_bottom).astype(int))
            return points
        if Break:
            break

import numpy as np

def draw_following_rectangle(frame, lines, prev_rect):
    rect_top_list = []
    rect_bottom_list = []
    if lines is not None:
        # Convertir les coordonnées des lignes en un tableau Numpy
        lines = np.array(lines)

        # Calculer la moyenne des coordonnées des lignes
        avg_line = np.mean(lines, axis=0, dtype=np.int32)

        # Calculer l'angle de la route
        x1, y1, x2, y2 = avg_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        # Calculer les coordonnées du rectangle suivant la direction de la route
        rect_center_x = (x1 + x2) // 2
        rect_center_y = (y1 + y2) // 2

        rect_width = 450   # Ajustez la largeur du rectangle selon vos besoins
        rect_height = 150  # Ajustez la hauteur du rectangle selon vos besoins

        rect_x1 = int(rect_center_x - rect_width // 2)
        rect_y1 = int(rect_center_y - rect_height // 2)
        rect_x2 = int(rect_x1 + rect_width)
        rect_y2 = int(rect_y1 + rect_height)

        # Faire la moyenne mobile avec les coordonnées du rectangle précédent
        alpha = 0.2  # Facteur de lissage (ajustez selon vos besoins)
        if prev_rect is not None:
            rect_x1 = int(alpha * rect_x1 + (1 - alpha) * prev_rect[0])
            rect_y1 = int(alpha * rect_y1 + (1 - alpha) * prev_rect[1])
            rect_x2 = int(alpha * rect_x2 + (1 - alpha) * prev_rect[2])
            rect_y2 = int(alpha * rect_y2 + (1 - alpha) * prev_rect[3])

        # Dessiner le rectangle sur l'image
        #cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)
        rect_top_list.append((rect_x1, rect_y1))
        rect_bottom_list.append((rect_x2, rect_y2))

        # Retourner les coordonnées du rectangle pour la mise à jour
        return rect_top_list,rect_bottom_list

    return prev_rect

def postprocessing(frame,model):
        import cv2
        import torch
        img = cv2.resize(frame,(640,640))# Resize the frame to match the expected input size

        # Perform inference
        with torch.no_grad():
                results = model(img)
        return results,img

def box_detections(bounding_box_diff, detections, center_cur_points, Class_points, frame, Conf_Pos, cls_dict):
    import numpy as np
    import cv2
    '''
    # Get boxes from results
    boxes = results.boxes  # assuming this returns a Boxes object
    if boxes is not None:

        # Extract the necessary data
        xyxy = boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
        print(xyxy)
        confs = boxes.conf.cpu().numpy()  # Get confidence scores
        clss = boxes.cls.cpu().numpy()  # Get class predictions
  
        for i in range(len(xyxy)):
            x_min, y_min, x_max, y_max = xyxy[i]
            L=[x_min, y_min, x_max, y_max]
            conf = confs[i]
            cls = int(clss[i]) #[{'class': 'person', 'conf': 0.87152821, 'box': array([ 138.30415344,  276.6373291 ,  626.50518799,  710.7076416 ], dtype=float32)}]
     '''
    if detections is not None:

        for i in range(len(detections)):
            label_text = detections[i]['class']
            if next((k for k,v in cls_dict.items() if v == label_text),None) is not None :
             if label_text=="person":
              box = detections[i]['box'].tolist()
              x_min, y_min, x_max, y_max = box[0],box[1],box[2],box[3]
              conf = detections[i]['conf']
              cls = int(next((k for k,v in cls_dict.items() if v == label_text),None))

              # Calculate the center of the bounding box
              cx = int((x_min + (x_max - x_min) / 2))
              cy = int((y_min + (y_max - y_min) / 2))

             # Determine the color based on the class
              if cls == 0:  # Person
                color = (0, 165, 255)  # Orange
              elif cls == 1:  # Bicycle
                color = (0, 0, 255)  # Red
              elif cls == 2:  # Car
                color = (255, 0, 0)  # Blue
              elif cls == 3:  # Motorcycle
                color = (128, 0, 128)  # Purple
              else:
                color = (0, 255, 0)  # Green by default if the class is not defined

              if conf >= 0.5:
                center_cur_points.append((cx, cy))

                Class_points[(cx, cy)] = cls
                Conf_Pos[(cx, cy)] = conf
                bounding_box_diff[(cx, cy)] = abs(y_max - y_min)
                '''
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                if cls in cls_dict:
                    label_text = f"{cls_dict[cls]}"
                    cv2.putText(frame, label_text, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                '''
    return center_cur_points, Conf_Pos, Class_points, bounding_box_diff



def tracking(frame_counter,center_cur_points,center_prev_points,Object_detection_list,Conf_Pos,Class_points,tracker_counter,id_counter,bounding_box_diff):
       if frame_counter == 1000:
               tracker_counter = 0
       if frame_counter <=2 : 
                                                                            
                                                    for pt in center_cur_points : 

                                                            for pt2 in center_prev_points:
                                                                distance = math.hypot(pt2[0] - pt [0], pt2[1] - pt[1]) 
                                                                if distance < 20 :
                                                                    if len(Object_detection_list) == 0:
                                                                        if pt in list(Conf_Pos) or pt in list(Class_points):
                                                                            Object_detection_list.append(Object_Detected(tracker_counter,pt,Conf_Pos[pt],Class_points[pt],bounding_box_diff[pt]))                                                                                                  
                                                                            tracker_counter +=1                                                                            
       else : 
                                                        for _object in Object_detection_list.copy():                                                        
                                                            object_exists = False                                                            
                                                            for pt in center_cur_points.copy():                                                            
                                                                distance = math.hypot(getattr(_object,'position')[0] - pt [0], getattr(_object,'position')[1] - pt[1])                                        
                                                                #Update object position
                                                                if distance <20:
                                                                    setattr(_object,'previous_position',getattr(_object,'position'))     #Register old polisition                                        
                                                                    setattr(_object,'position',pt)
                                                                    object_exists = True
                                                                    if pt in center_cur_points:
                                                                        center_cur_points.remove(pt)
                                                                    continue
                                                            if not object_exists :
                                                                    Object_detection_list.remove(_object)

                                                        for pt in center_cur_points:   
                                                            #for object_ in Object_detection_list.copy() :
                                                                if pt in list(Conf_Pos) or pt in list(Class_points):
                                                                            Object_detection_list.append(Object_Detected(tracker_counter,pt,Conf_Pos[pt],Class_points[pt],bounding_box_diff[pt]))                                                
                                                                            tracker_counter +=1     
                                                        for obj in Object_detection_list : 
                                                                id_counter.append(getattr(obj,'id'))                 
       return Object_detection_list,id_counter,tracker_counter


def video_vizualisation(agreg,frame,send_data_var,send_data_counter,data_send):
                            cars = len(getattr(getattr(agreg,'counter'),'car_counter'))
                            pedestrians = len(getattr(getattr(agreg,'counter'),'pedestrian_counter'))
                            bus = len(getattr(getattr(agreg,'counter'),'bus_counter'))
                            truck = len(getattr(getattr(agreg,'counter'),'truck_counter'))
                            moto = len(getattr(getattr(agreg,'counter'),'moto_counter'))
                            bike = len(getattr(getattr(agreg,'counter'),'bicycle_counter'))

                            if data_send : 
                                send_data_var  +=1
                                cv2.putText(frame, f'CARS {cars}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)  
                                cv2.putText(frame, f'PEDESTRIANS {pedestrians}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
                                cv2.putText(frame, f'BUS {bus}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)  
                                cv2.putText(frame, f'TRUCKS {truck}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255),2)  
                                cv2.putText(frame, f'MOTOS {moto}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)  
                                cv2.putText(frame, f'BIKES {bike}', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                            else :  
                                cv2.putText(frame, f'CARS {cars}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50),2)  
                                cv2.putText(frame, f'PEDESTRIANS {pedestrians}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50),2)  
                                cv2.putText(frame, f'BUS {bus}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50),2)  
                                cv2.putText(frame, f'TRUCKS {truck}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50),2)  
                                cv2.putText(frame, f'MOTOS {moto}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50),2)  
                                cv2.putText(frame, f'BIKES {bike}', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50),2)  
                            
                            if send_data_var == send_data_counter+ 10 :
                                   data_send = False
                                   send_data_counter = 0
                                   send_data_var = 0


def remove_from_counter(agreg,id_counter):
                            for _id in getattr(getattr(agreg,'counter'),'car_counter'): 
                                                if _id not in id_counter : 
                                                            getattr(agreg,'counter').remove_car(_id) 

                            for _id in getattr(getattr(agreg,'counter'),'pedestrian_counter') : 
                                                     if _id not in id_counter : 
                                                            getattr(agreg,'counter').remove_pedestrian(_id)    
                            for _id in getattr(getattr(agreg,'counter'),'bus_counter') : 
                                                     if _id not in id_counter : 
                                                           getattr(agreg,'counter').remove_bus(_id)  
                            for _id in getattr(getattr(agreg,'counter'),'truck_counter') : 
                                                     if _id not in id_counter : 
                                                            getattr(agreg,'counter').remove_truck(_id)          
                            for _id in getattr(getattr(agreg,'counter'),'moto_counter') : 
                                                    if _id not in id_counter : 
                                                            getattr(agreg,'counter').remove_moto(_id) 
                            for _id in getattr(getattr(agreg,'counter'),'bicycle_counter') : 
                                                     if _id not in id_counter : 
                                                            getattr(agreg,'counter').remove_bicycle(_id)


def send_counter_data_its(agreg,myobj):
                                import requests
                                access_token = get_token()

                                url = 'https://its.labatosbordeaux.eu/back/cdpits/aggregationRequest/data'

                                headers = {'Authorization': f'Bearer {access_token}'}
 
                                requests.post(url, json = myobj, headers = headers, verify=False)

                                
                                print(myobj)
