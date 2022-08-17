import cv2 as cv
import time
import numpy as np

# nombres de las clases
with open('./input/coco_classes.txt','r') as f:
    class_names = f.read().split('\n')

# cargar el modelo
model = cv.dnn.readNet( model='./input/frozen_inference_graph.pb',
                        config='./input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')

# generar un color para cada clase
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# señal de video
vid = cv.VideoCapture(0)

# obtener detalles de la señal de video
# altura y anchura de cada cuadro
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
# objeto VideoWriter()
out = cv.VideoWriter('./video_result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 
                    30, (frame_width, frame_height))
    
# procesar la señal de video
while(True):
    ret, vidFrame = vid.read()
    if ret:
        image = vidFrame
        image_height, image_width, _ = image.shape
        # procesar la imagen
        blob = cv.dnn.blobFromImage(image = image, size=(300,300), 
                                    mean=(104, 117, 123), swapRB=True)
        
        # iniciar tiempo para calcular fps
        start = time.time()
        model.setInput(blob)
        output = model.forward()

        # ciclar las detecciones
        for detection in output[0, 0, :, :]:
            # sacar la confianza
            confidence = detection[2]
            # dibujar las cajas
            if confidence > .4:
                # id de clase
                class_id = detection[1]
                # relacionar id con la clase
                class_name = class_names[int(class_id)-1]
                color = COLORS[int(class_id)]
                # coordenadas de la caja
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                # dimensiones de la caja
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                # dibujar las cajas
                cv.rectangle(image, (int(box_x), int(box_y)), (int(box_width), 
                            int(box_height)), color, thickness=2)
                # colocar el nombre de la clase
                cv.putText(image, class_name, (int(box_x), int(box_y - 5)), 
                            cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                # poner el contador de fps
                cv.putText(image, f"{fps:.2f} FPS", (20, 30), 
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        cv.imshow('video', image)
        out.write(image)
        
        # terminar tiempo
        end = time.time()
        # calculo de fps
        fps = 1 / (end-start)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

vid.release()
cv.destroyAllWindows()
