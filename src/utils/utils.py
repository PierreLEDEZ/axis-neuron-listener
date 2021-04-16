CLIPLIMIT = 10
SCALE_FACTOR=1.05

IGNORED_JOINTS = [
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightInHandIndex",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightInHandMiddle",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightInHandRing",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightInHandPinky",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftInHandIndex",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftInHandMiddle",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftInHandRing",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftInHandPinky",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3"
]

CLASSES = [
    "assemble_system",
    "consult_sheets",
    "picking_in_front",
    "picking_left",
    "put_down_component",
    "put_down_measuring_rod",
    "put_down_screwdriver",
    "put_down_subsystem",
    "take_component",
    "take_measuring_rod",
    "take_screwdriver",
    "take_subsystem",
    "turn_sheets"
]

def ignoreJoints(bvhParser, coordinate="rgb", ignored_joints=[]):
    """ 
        Find joint indexes of ignored joints.
        If both coordinates are used (translation and rotation), indexes are doubled

        :param bvhParser: bvhParser object
        :type bvhParser: BVHParser Class
        :param coordinate: coordinates used to convert bvh to image
        :type path: string
        :param ignored_joints: names of ignored joints
        :type filename: array
        
        :return: indexes corresponding to the list of ignored joints 
        :rtype: array
    """

    if coordinate == "rgb":
        ignored_joints_index_pos = [2*bvhParser.joints[j].index for j in ignored_joints]
        ignored_joints_index_rot = [2*bvhParser.joints[j].index+1 for j in ignored_joints]
        ignored_joints_index = [*ignored_joints_index_pos, *ignored_joints_index_rot]
    else:
        ignored_joints_index = [bvhParser.joints[j].index for j in ignored_joints]
    
    return ignored_joints_index

def get_min_max_rotation(frames):
    """ 
        Get the minimum and maximum of rotation coordinates for the 3 axis (x, y, z) in all the frames 

        :param frames: The frames
        :type frames: numpy array 
        
        :return: Three tuples (one per axis) with the minimum and maximum rotation coordinates
        :rtype: tuple(float, float)
    """

    X_rotation = []
    Y_rotation = []
    Z_rotation = []

    for _, frame in enumerate(frames):
        for joint in range(0, len(frame), 6):
            for pixel in range(0, 3):
                try:
                    value = frame[joint + 3 + pixel]
                except IndexError:
                    continue
                if pixel == 0:
                    Y_rotation.append(value)
                elif pixel == 1:
                    X_rotation.append(value)
                else:
                    Z_rotation.append(value)

    return (min(X_rotation), max(X_rotation)), (min(Y_rotation), max(Y_rotation)), (min(Z_rotation), max(Z_rotation))

def get_min_max_translation(frames):
    """ 
        Get the minimum and maximum of translation coordinates for the 3 axis (x, y, z) in all the frames 

        :param frames: The frames
        :type frames: numpy array 
        
        :return: Three tuples (one per axis) with the minimum and maximum translation coordinates
        :rtype: tuple(float, float)
    """
    
    X_translation = []
    Y_translation = []
    Z_translation = []

    for _, frame in enumerate(frames):
        for joint in range(0, len(frame), 6):
            for pixel in range(0, 3):
                try:
                    value = frame[joint + pixel]
                except IndexError:
                    continue
                if pixel == 0:
                    X_translation.append(value)
                elif pixel == 1:
                    Y_translation.append(value)
                else:
                    Z_translation.append(value)

    return (min(X_translation), max(X_translation)), (min(Y_translation), max(Y_translation)), (min(Z_translation), max(Z_translation))

def calculate_laraba_value_rotation(frame, joint, min_max_X, min_max_Y, min_max_Z):
    """ 
        Calculate RGB values with Laraba's method for rotation coordinates 

        :param frame: The current frame
        :type frame: numpy array 
        :param joint: The current joint index
        :type joint: int
        :param min_max_X: Minimum and maximum rotation coordinates around X axis
        :type min_max_X: tuple(float, float)
        :param min_max_Y: Minimum and maximum rotation coordinates around Y axis
        :type min_max_Y: tuple(float, float)
        :param min_max_Z: Minimum and maximum rotation coordinates around Z axis
        :type min_max_Z: tuple(float, float)
        
        :return: RGB values corresponding respectively to the x, y and z rotation coordinates
        :rtype: array[float, float, float]
    """

    # Read first the coordinate at indice 1 because rotations are stored in the order: YXZ 
    return [
        255 * ((frame[joint + 1] - min_max_X[0]) / (min_max_X[1] - min_max_X[0])),
        255 * ((frame[joint] - min_max_Y[0]) / (min_max_Y[1] - min_max_Y[0])),
        255 * ((frame[joint + 2]- min_max_Z[0]) / (min_max_Z[1] - min_max_Z[0]))
    ]

def calculate_laraba_value_translation(frame, joint, min_max_X, min_max_Y, min_max_Z):
    """ 
        Calculate RGB values with Laraba's method for translation coordinates 

        :param frame: The current frame
        :type frame: numpy array 
        :param joint: The current joint index
        :type joint: int
        :param min_max_X: Minimum and maximum translation coordinates along X axis
        :type min_max_X: tuple(float, float)
        :param min_max_Y: Minimum and maximum translation coordinates along Y axis
        :type min_max_Y: tuple(float, float)
        :param min_max_Z: Minimum and maximum translation coordinates along Z axis
        :type min_max_Z: tuple(float, float)
        
        :return: RGB values corresponding respectively to the x, y and z translation coordinates
        :rtype: array[float, float, float]
    """

    return [
        255 * (((frame[joint] * SCALE_FACTOR) - min_max_X[0]) / (min_max_X[1] - min_max_X[0])),
        255 * (((frame[joint + 1] * SCALE_FACTOR) - min_max_Y[0]) / (min_max_Y[1] - min_max_Y[0])),
        255 * (((frame[joint + 2] * SCALE_FACTOR) - min_max_Z[0]) / (min_max_Z[1] - min_max_Z[0]))
    ]