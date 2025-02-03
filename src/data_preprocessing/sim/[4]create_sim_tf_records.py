import os
import sys

sys.path.insert(0, os.getcwd())
import src.models.DeepProbCP.tfrecords_handler.tfrecord_writer as tw

output_path = "./data/sim/binary_data/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

input_size = 12 
output_size = 24


if __name__ == '__main__':
    seq_lens = [90,420]
    no_series = [50,300]
    trend = ['stationary','trend']
    interv_type = ['hom'] 
    pred_len = 24

    for seq in seq_lens:
        for nseries in no_series:
            for tr in trend:
                 for interv in interv_type:
                    tfrecord_writer = tw.TFRecordWriter(
                        input_size = input_size +1 ,
                        output_size = output_size,
                        train_file_path = './data/sim/moving_window/' + 'sim' + '_' + str(seq) + '_' + str(nseries) + '_' + str(tr) + '_' + \
                           str(interv) + '_' + str(input_size) + '_' + str(output_size) + '_train' + '.txt',
                        validate_file_path = './data/sim/moving_window/' + 'sim' + '_' + str(seq) + '_' + str(nseries) + '_' + str(tr) + '_' + \
                            str(interv) + '_' + str(input_size) + '_' + str(output_size) + '_val' + '.txt',
                        test_file_path = './data/sim/moving_window/' + 'sim' + '_' + str(seq) + '_' + str(nseries) + '_' + str(tr) + '_' + \
                            str(interv) + '_' + str(input_size) + '_' + str(output_size) + '_test' + '.txt', 
                        binary_train_file_path = output_path + 'sim' + '_' + str(seq) + '_' + str(nseries) + '_' + str(tr) + '_' + \
                           str(interv) + '_' + str(input_size) + '_' + str(output_size) + '_train' +  '.tfrecords',
                        binary_validation_file_path = output_path + 'sim' + '_' + str(seq) + '_' + str(nseries) + '_' + str(tr) + '_' + \
                            str(interv) + '_' + str(input_size) + '_' + str(output_size) + '_val' + '.tfrecords',
                        binary_test_file_path = output_path + 'sim' + '_' + str(seq) + '_' + str(nseries) + '_' + str(tr) + '_' + \
                            str(interv) + '_' + str(input_size) + '_' + str(output_size) + '_test' + '.tfrecords'
                    )

                    tfrecord_writer.read_text_data()
                    tfrecord_writer.write_train_data_to_tfrecord_file()
                    tfrecord_writer.write_validation_data_to_tfrecord_file()
                    tfrecord_writer.write_test_data_to_tfrecord_file()
