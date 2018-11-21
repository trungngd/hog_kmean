import os, numpy as np, matplotlib.pyplot as plt, cv2

root_dir = '/media/data/datasets/Kinect2017-10/Datasets/'
out_dir = '/home/nguyenductrung/kinect_vis/output/'


# out_dir='/media/data/datasets/Kinect_vis/v2/'
def read_skeleton(path):
    data = {}
    f = open(path, 'rt')
    while True:
        line = f.readline()
        if line == '':
            break
        frame_idx = int(line.split(' ')[2])
        num_ppl = int(line.split(' ')[3])

        skels = []
        for i in range(num_ppl):
            line = f.readline().split(' ')
            skel = []
            for j in range(20):
                skel.append([int(float(line[j * 3])), int(float(line[j * 3 + 1]))])
            skels.append(skel)
        # print skel
        # break
        data[frame_idx] = skels

    # print line
    return data


# for _, dirs, _ in os.walk(root_dir):
#     break
dirs = ['20171123_Phong_lan2_23-11-2017__11-49-53']
print dirs
for f_name in dirs:
    # f_name='20171123_Hung_lan1_23-11-2017__11-05-57'
    vid_dir = root_dir + f_name + '/'
    out_vid_path = out_dir + f_name + '.avi'
    if os.path.isfile(out_vid_path):
        continue
    print f_name


    def read_acc(file_name):
        f = open(vid_dir + file_name, 'rt')
        lines = f.read().strip().split('\n')
        acc_hand = []
        # print lines[:10]
        for l in lines:
            row = [float(v) for v in l.split(' ')]
            if acc_hand == [] or (row[1] > acc_hand[-1][1]) or row[1] == 0:
                # print row
                acc_hand.append(row)

        # print len(acc_hand)
        start_timestamp = acc_hand[0][0]
        end_timestamp = acc_hand[-1][0]

        duration = end_timestamp - start_timestamp  # ms
        # print duration
        acc_new = np.zeros((int(duration) / 10 + 1, 3), dtype=float)

        for acc in acc_hand:
            time_diff = int(acc[0] - start_timestamp) / 10
            # print time_diff
            # print acc[1]
            acc_new[time_diff, 0] = acc[2]
            acc_new[time_diff, 1] = acc[3]
            acc_new[time_diff, 2] = acc[4]
        return acc_new


    import struct


    # from scipy.ndimage.interpolation import shift

    def read_depth(file_reader):
        try:
            byte = file_reader.read(4)
            val = struct.unpack('=L', byte)
            # print 'img size: '+str(val[0])
            img_data = np.fromstring(file_reader.read(val[0]), dtype=np.uint8)
            depth = cv2.imdecode(img_data, cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (640 / 3, 480 / 3))
            depth = np.right_shift(depth, 7)
            depth = np.asarray(depth, np.uint8)

            depth = 255 - depth
            # ret, depth = cv2.threshold(depth, 254, 255, cv2.THRESH_TOZERO_INV)

            depth_vis = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            # depth
            return depth_vis
        except Exception as e:
            print 'Error: ' + str(e)
            return None


    # each row is data for 10 ms
    acc_hand = read_acc('1.txt')
    acc_belt = read_acc('155.txt')

    # print acc_hand[:100]

    duration = acc_hand.shape[0] / 100.0  # s
    n_frames = int(duration * 20.0)  # each frame is 50ms

    pixels_per_sec = 50  # width
    pixels_per_ms = pixels_per_sec / 1000.0  # width
    length_acc_on_img = 1180 / 50  # 10s

    color_vids = []
    depth_vids = []
    skeleton_all = []
    for i in range(1, 8):
        vid = cv2.VideoCapture(vid_dir + '/Kinect_' + str(i) + '/color.avi')
        color_vids.append(vid)
        print vid_dir + 'Kinect_' + str(i) + '/depth.bin'
        f = open(vid_dir + 'Kinect_' + str(i) + '/depth.bin', "rb")
        depth_vids.append(f)
        skeleton_all.append(read_skeleton(vid_dir + '/Kinect_' + str(i) + '/skeleton.txt'))

    writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'H264'), 20.0, (1920, 1440))

    frame_idx = 0
    # cv2.namedWindow('',cv2.WINDOW_NORMAL)
    try:
        while True:
            if frame_idx % 10 == 0:
                print frame_idx
            img_total = np.zeros((480 * 3, 640 * 3, 3), np.uint8)
            for kid in range(7):
                ret, img = color_vids[kid].read()
                if img is None:
                    break
                depth = read_depth(depth_vids[kid])
                # print 'Color shape is:'+str(img.shape)
                # print 'depth shape is:'+str(depth.shape)
                location_x = kid % 3 * 640
                location_y = kid / 3 * 480

                # draw skeleton
                # if frame_idx in skeleton_all[kid]:
                #     skels = skeleton_all[kid][frame_idx]
                #
                #     for skel in skels:
                #         # print skel
                #
                #         for k in range(3):
                #             cv2.line(img, (skel[k][0], skel[k][1]), (skel[k + 1][0], skel[k + 1][1]), (0, 255, 0), 2)
                #             cv2.line(img, (skel[k + 4][0], skel[k + 4][1]), (skel[k + 4 + 1][0], skel[k + 4 + 1][1]),
                #                      (0, 255, 0), 2)
                #             cv2.line(img, (skel[k + 8][0], skel[k + 8][1]), (skel[k + 8 + 1][0], skel[k + 8 + 1][1]),
                #                      (0, 255, 0), 2)
                #             cv2.line(img, (skel[k + 12][0], skel[k + 12][1]),
                #                      (skel[k + 12 + 1][0], skel[k + 12 + 1][1]), (0, 255, 0), 2)
                #             cv2.line(img, (skel[k + 16][0], skel[k + 16][1]),
                #                      (skel[k + 16 + 1][0], skel[k + 16 + 1][1]), (0, 255, 0), 2)
                #         k = 0
                #         cv2.line(img, (skel[k][0], skel[k][1]), (skel[k + 12][0], skel[k + 12][1]), (0, 255, 0), 2)
                #         cv2.line(img, (skel[k][0], skel[k][1]), (skel[k + 16][0], skel[k + 16][1]), (0, 255, 0), 2)
                #         cv2.line(img, (skel[k + 2][0], skel[k + 2][1]), (skel[k + 4][0], skel[k + 4][1]), (0, 255, 0),
                #                  2)
                #         cv2.line(img, (skel[k + 2][0], skel[k + 2][1]), (skel[k + 8][0], skel[k + 8][1]), (0, 255, 0),
                #                  2)
                #         for i in range(20):
                #             cv2.circle(img, (skel[i][0], skel[i][1]), 3, (0, 0, 255), -1)

                img_total[location_y:location_y + 480, location_x:location_x + 640, :] = img
                img_total[location_y:location_y + 160, location_x:location_x + 213, :] = depth

                cv2.putText(img_total, str(kid + 1), (location_x + 10, location_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 5, cv2.LINE_AA)
            # print depth.shape
            # break
            if img is None:
                break
            img = img_total

            # end_acc_idx = frame_idx * 5
            # begin_acc_idx = end_acc_idx - length_acc_on_img * 100
            # if begin_acc_idx < 0:
            #     begin_acc_idx = 0
            #
            # prev_pt = [0.0, 0.0, 0.0, 0.0]
            # for j in range(begin_acc_idx, end_acc_idx)[::-1]:
            #     acc_x = acc_hand[j]
            #     location_x = 1980 - 100 - int((end_acc_idx - j) * 10 * pixels_per_ms)
            #     if acc_x[0] != 0.0 and acc_x[1] != 0.0 and acc_x[2] != 0.0:
            #         acc_location_x = int((acc_x[0] + 1) * 30 + 1100)
            #         acc_location_y = int((acc_x[1] + 1) * 30 + 1100)
            #         acc_location_z = int((acc_x[2] + 1) * 30 + 1100)
            #         curr_pt = [location_x, acc_location_x, acc_location_y, acc_location_z]
            #         if prev_pt[0] != 0.0 and prev_pt[1] != 0.0 and prev_pt[2] != 0.0 and prev_pt[3] != 0.0:
            #             cv2.line(img, (prev_pt[0], prev_pt[1]), (curr_pt[0], curr_pt[1]), (255, 0, 0), 2)
            #             cv2.line(img, (prev_pt[0], prev_pt[2]), (curr_pt[0], curr_pt[2]), (0, 255, 0), 2)
            #             cv2.line(img, (prev_pt[0], prev_pt[3]), (curr_pt[0], curr_pt[3]), (0, 0, 255), 2)
            #         # break
            #         prev_pt = curr_pt
            #
            # prev_pt = [0.0, 0.0, 0.0, 0.0]
            # for j in range(begin_acc_idx, end_acc_idx)[::-1]:
            #     acc_x = acc_belt[j]
            #     location_x = 1980 - 100 - int((end_acc_idx - j) * 10 * pixels_per_ms)
            #     if acc_x[0] != 0.0 and acc_x[1] != 0.0 and acc_x[2] != 0.0:
            #         acc_location_x = int((acc_x[0] + 1) * 30 + 1300)
            #         acc_location_y = int((acc_x[1] + 1) * 30 + 1300)
            #         acc_location_z = int((acc_x[2] + 1) * 30 + 1300)
            #         curr_pt = [location_x, acc_location_x, acc_location_y, acc_location_z]
            #         if prev_pt[0] != 0.0 and prev_pt[1] != 0.0 and prev_pt[2] != 0.0 and prev_pt[3] != 0.0:
            #             cv2.line(img, (prev_pt[0], prev_pt[1]), (curr_pt[0], curr_pt[1]), (255, 0, 0), 2)
            #             cv2.line(img, (prev_pt[0], prev_pt[2]), (curr_pt[0], curr_pt[2]), (0, 255, 0), 2)
            #             cv2.line(img, (prev_pt[0], prev_pt[3]), (curr_pt[0], curr_pt[3]), (0, 0, 255), 2)
            #         # break
            #         prev_pt = curr_pt
            # plt.imshow(img)
            # plt.show()
            # break
            writer.write(img)
            # cv2.namedWindow('',cv2.WINDOW_NORMAL)
            # cv2.imshow('',img)
            # if cv2.waitKey(10)==27:
            # 	break
            # break
            # pass

            # cv2.imshow('',img)
            # cv2.waitKey(10)
            frame_idx += 1
            pass
    except Exception as e:
        pass
    for i in range(7):
        depth_vids[i].close()

# writer.close()