from __future__ import print_function

cimport quanser_types as qt
cimport numpy as np
cimport hil

from gym_brt.envs.QuanserWrapper.helpers.error_codes import print_possible_error

from threading import Thread, Lock
import numpy as np
import time


cdef class QuanserWrapper:
    cdef hil.t_card _board
    cdef hil.t_task _task

    cdef qt.t_uint32[::] _analog_r_channels
    cdef qt.t_uint32 _num_analog_r_channels
    cdef qt.t_double[::] _currents_r

    cdef qt.t_uint32[::] _analog_w_channels
    cdef qt.t_uint32 _num_analog_w_channels
    cdef qt.t_double[::] _voltages_w

    cdef qt.t_uint32[::] _digital_w_channels
    cdef qt.t_uint32 _num_digital_w_channels
    cdef qt.t_boolean[::] _enables_r

    cdef qt.t_uint32[::] _encoder_r_channels
    cdef qt.t_uint32 _num_encoder_r_channels
    cdef qt.t_int32[::] _encoder_r_buffer

    cdef qt.t_uint32[::] _other_r_channels
    cdef qt.t_uint32 _num_other_r_channels
    cdef qt.t_double[::] _other_r_buffer

    cdef qt.t_uint32[::] _led_w_channels
    cdef qt.t_uint32 _num_led_w_channels
    cdef qt.t_double[::] _led_w_buffer

    cdef qt.t_double _frequency, _safe_operating_voltage

    cdef float _last_read_time
    cdef bint _task_started, _new_state_read
    cdef object _bg_thread, _lock
    cdef int _num_samples_read_since_action

    def __init__(self,
                 safe_operating_voltage,
                 analog_r_channels,
                 analog_w_channels,
                 digital_w_channels,
                 encoder_r_channels,
                 other_r_channels,
                 led_w_channels,
                 frequency=1000):
        """
        Args:
            - safe_operating_voltage: The largest allowed voltage as an output
                (motor voltage)
            - analog_r_channels: [INPUT] a list of analog channels to use for
                commumication
            - analog_w_channels: [OUTPUT] a list of analog channels to use for
                commumication
            - digital_w_channels: [INPUT] a list of digital channels to use for
                commumication
            - encoder_r_channels: [INPUT] a list of encoder channels to use for
                commumication
            - other_r_channels: [INPUT] a list of other channels to use for
                commumication
            - led_w_channels: [OUTPUT] a list of led channels to use for
                commumication
            - Frequency: Frequency of the reading/writing task (in Hz)
        """
        self._safe_operating_voltage = safe_operating_voltage
        # Convert the channels into numpy arrays which are then stored in 
        # memoryviews (to pass C buffers to the HIL API)
        self._num_analog_r_channels = len(analog_r_channels)
        self._num_analog_w_channels = len(analog_w_channels)
        self._num_digital_w_channels = len(digital_w_channels)
        self._num_encoder_r_channels = len(encoder_r_channels)
        self._num_other_r_channels = len(other_r_channels)
        self._num_led_w_channels = len(led_w_channels)
        self._analog_r_channels = np.array(analog_r_channels, dtype=np.uint32)
        self._analog_w_channels = np.array(analog_w_channels, dtype=np.uint32)
        self._digital_w_channels = np.array(digital_w_channels, dtype=np.uint32)
        self._encoder_r_channels = np.array(encoder_r_channels, dtype=np.uint32)
        self._other_r_channels = np.array(other_r_channels, dtype=np.uint32)
        self._led_w_channels = np.array(led_w_channels, dtype=np.uint32)

        self._lock = Lock()
        self._frequency = frequency
        self._task_started = False
        self._new_state_read = False
        self._num_samples_read_since_action = 0
        self._bg_thread = Thread(target=self.run_reader_writer, args=())
        self._last_read_time = 0

    def __enter__(self):
        """
        Start the hardware in a deterministic way (all motors, encoders, etc
        at 0)
        """
        # Create a memoryview for currents
        self._currents_r = np.zeros(
            self._num_analog_r_channels, dtype=np.float64)  # t_double is 64 bits

        # Create a memoryview for -ometers
        self._other_r_buffer = np.zeros(
            self._num_other_r_channels, dtype=np.float64)  # t_double is 64 bits

        # Create a memoryview for leds
        self._led_w_buffer = np.zeros(
            self._num_led_w_channels, dtype=np.float64)  # t_double is 64 bits

        # Set all motor voltages_w to 0
        self._voltages_w = np.zeros(
            self._num_analog_w_channels, dtype=np.float64)  # t_double is 64 bits
        result = hil.hil_write_analog(
            self._board,
            &self._analog_w_channels[0],
            self._num_analog_w_channels,
            &self._voltages_w[0])
        print_possible_error(result)

        # Set the encoder encoder_r_buffer to 0
        self._encoder_r_buffer = np.zeros(
            self._num_encoder_r_channels, dtype=np.int32)  # t_int32 is 32 bits
        result = hil.hil_set_encoder_counts(
            self._board,
            &self._encoder_r_channels[0],
            self._num_encoder_r_channels,
            &self._encoder_r_buffer[0])
        print_possible_error(result)

        # Enables_r all the motors
        self._enables_r = np.ones(
            self._num_digital_w_channels, dtype=np.int8)  # t_bool is char 8 bits
        result = hil.hil_write_digital(
            self._board,
            &self._digital_w_channels[0],
            self._num_digital_w_channels,
            &self._enables_r[0])
        print_possible_error(result)

        return self

    def __exit__(self, type, value, traceback):
        """Make sure hardware turns off safely"""
        self._stop_task()

        # Set the motor voltages_w to 0
        self._voltages_w = np.zeros(
            self._num_analog_w_channels, dtype=np.float64)  # t_double is 64 bits
        hil.hil_write_analog(
            self._board,
            &self._analog_w_channels[0],
            self._num_analog_w_channels,
            &self._voltages_w[0])

        # Disable all the motors
        self._enables_r = np.zeros(
            self._num_digital_w_channels, dtype=np.int8)  # t_bool is char 8 bits
        hil.hil_write_digital(
            self._board,
            &self._digital_w_channels[0],
            self._num_digital_w_channels,
            &self._enables_r[0])

        hil.hil_close(self._board)  # Safely close the board

    def _create_task(self):
        """Start a task reads and writes at fixed intervals"""
        result =  hil.hil_task_create_reader(
            self._board,
            1,  # The size of the internal buffer (making this >> 1 
                # prevents error 111 but may also occasionally miss a read
                # of state)
            &self._analog_r_channels[0], self._num_analog_r_channels,
            &self._encoder_r_channels[0], self._num_encoder_r_channels,
            NULL, 0,
            &self._other_r_channels[0], self._num_other_r_channels,
            &self._task)
        print_possible_error(result)

        # Start the task
        result = hil.hil_task_start(
            self._task,
            hil.HARDWARE_CLOCK_0,
            self._frequency,
            -1) # Read continuously 
        print_possible_error(result)
        if result < 0:
            raise ValueError("Could not start hil task")

        self._task_started = True
        self._bg_thread.start()

    def _stop_task(self):
        if self._task_started:
            self._task_started = False
            self._bg_thread.join()
            hil.hil_task_flush(self._task)
            hil.hil_task_stop(self._task)
            hil.hil_task_delete(self._task)

    def run_reader_writer(self):
        """Helper function to pass as a python callable to `Thread`"""
        self._run_reader_writer()

    cdef _run_reader_writer(self):
        """Run background thread that continously updates QuanserWrapper's
        internal buffers with the newest state at a sample instant, and writes
        the current action buffer to the board.
        """
        cdef hil.t_error samples_read, result_write
        cdef qt.t_double[::] temp_currents_r = np.empty_like(self._currents_r)
        cdef qt.t_int32[::] temp_encoder_r_buffer = np.empty_like(
            self._encoder_r_buffer)
        cdef qt.t_double[::] temp_other_r_buffer = np.empty_like(
            self._other_r_buffer)

        while self._task_started:
            # First read using task_read (blocking call that enforces timing)
            samples_read = hil.hil_task_read(
                self._task,
                1, # Number of samples to read
                &temp_currents_r[0],
                &temp_encoder_r_buffer[0],
                NULL,
                &temp_other_r_buffer[0])
            if samples_read < 0:
                print_possible_error(samples_read)

            with self._lock:
                # Copy the temp state buffers into the quanser wrapper buffers
                self._currents_r = temp_currents_r
                self._encoder_r_buffer = temp_encoder_r_buffer
                self._other_r_buffer = temp_other_r_buffer

                # Then write voltages_w calculated for previous time step
                result_write = hil.hil_write_analog(
                    self._board,
                    &self._analog_w_channels[0],
                    self._num_analog_w_channels,
                    &self._voltages_w[0])
                if result_write < 0:
                    print_possible_error(result_write)

                self._new_state_read = True
                self._num_samples_read_since_action += 1

            time.sleep(0.1 / self._frequency)

    def action(self, voltages_w):
        """Make sure you get safe data!"""    
        # If it's the first time running action, then start the background r/w 
        # task
        if not self._task_started:
            self._create_task()

        if isinstance(voltages_w, list):
            voltages_w = np.array(voltages_w, dtype=np.float64)
        assert isinstance(voltages_w, np.ndarray)
        assert voltages_w.shape == (self._num_analog_w_channels,)
        assert voltages_w.dtype == np.float64
        for i in range(self._num_analog_w_channels):
            assert -self._safe_operating_voltage <= voltages_w[i] <= \
                    self._safe_operating_voltage

        self._action(voltages_w)
        self._action(voltages_w)
        self._action(voltages_w)
        return self._action(voltages_w)

    def _action(self,
                np.ndarray[qt.t_double, ndim=1, mode="c"] voltages_w not None):
        """Perform actions on the device (voltages_w must always be ndarray!)"""

        with self._lock:
            # Print warning if buffer read has been missed
            if self._num_samples_read_since_action > 1:
                print("Warning:", self._num_samples_read_since_action - 1,
                      "samples have been missed since last env step")

            # Update the action in the quanser wrapper buffer
            self._voltages_w = voltages_w.copy()

        while True:
            # Make sure to get the most recent state from the background reader
            time.sleep(0.1 / self._frequency)
            with self._lock:
                if self._new_state_read:
                    currents_r = np.asarray(self._currents_r).copy()
                    encoder_r_buffer = np.asarray(self._encoder_r_buffer).copy()
                    other_r_buffer = np.asarray(self._other_r_buffer).copy()
                    self._num_samples_read_since_action = 0
                    self._new_state_read = False
                    break
        return currents_r, encoder_r_buffer, other_r_buffer


cdef class QuanserAero(QuanserWrapper):
    def __cinit__(self):
        board_type = b"quanser_aero_usb"
        board_identifier = b"0"
        result = hil.hil_open(board_type, board_identifier, &self._board)
        print_possible_error(result)
        if result < 0:
            raise IOError("Board could not be opened.")

    def __init__(self, frequency=100):
        analog_r_channels = [0, 1]
        analog_w_channels = [0, 1]
        digital_w_channels = [0, 1]
        encoder_r_channels = [0, 1, 2, 3]
        other_r_channels = [3000, 3001, 3002, 4000, 4001, 4002, 14000, 14001, \
                             14002, 14003]
        led_w_channels = [11000, 11001, 11002]

        super(QuanserAero, self).__init__(
            safe_operating_voltage=18.0,
            analog_r_channels=analog_r_channels,
            analog_w_channels=analog_w_channels,
            digital_w_channels=digital_w_channels,
            encoder_r_channels=encoder_r_channels,
            other_r_channels=other_r_channels,
            led_w_channels=led_w_channels,
            frequency=frequency)


cdef class QubeServo2(QuanserWrapper):
    def __cinit__(self):
        board_type = b"qube_servo2_usb"
        board_identifier = b"0"
        result = hil.hil_open(board_type, board_identifier, &self._board)
        print_possible_error(result)
        if result < 0:
            raise IOError("Board could not be opened.")

    def __init__(self, frequency=100):
        analog_r_channels = [0]
        analog_w_channels = [0]
        digital_w_channels = [0]
        encoder_r_channels = [0, 1]
        other_r_channels = [14000]
        led_w_channels = [11000, 11001, 11002]

        super(QubeServo2, self).__init__(
            safe_operating_voltage=18.0,
            analog_r_channels=analog_r_channels,
            analog_w_channels=analog_w_channels,
            digital_w_channels=digital_w_channels,
            encoder_r_channels=encoder_r_channels,
            other_r_channels=other_r_channels,
            led_w_channels=led_w_channels,
            frequency=frequency)
