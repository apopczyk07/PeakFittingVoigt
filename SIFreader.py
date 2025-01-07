import numpy as np

class SIFReader:
    def __init__(self, file):
        self.file = file

    def read_sif(self):
        with open(self.file, 'rb') as f:
            if not self._read_line(f).startswith('Andor Technology Multi-Channel File'):
                raise ValueError('Not an Andor SIF image file.')

            self._skip_lines(f, 1)
            data = self._read_section(f)
        return data

    def _read_section(self, f):
        data = {}

        # Parse temperature and initial header information
        o = self._read_integers(f, 6)
        data['temperature'] = o[5]
        self._skip_bytes(f, 10)

        # Parse additional metadata
        o = self._read_floats(f, 5)
        data['delayExpPeriod'] = o[1]
        data['exposureTime'] = o[2]
        data['accumulateCycles'] = o[4]
        data['accumulateCycleTime'] = o[3]
        self._skip_bytes(f, 2)

        o = self._read_floats(f, 2)
        data['stackCycleTime'] = o[0]
        data['pixelReadoutTime'] = o[1]

        o = self._read_integers(f, 3)
        data['gainDAC'] = o[2]
        self._skip_lines(f, 1)

        data['detectorType'] = self._read_line(f).strip()
        data['detectorSize'] = self._read_integers(f, 2)
        data['fileName'] = self._read_string(f)

        # Skip until specific markers and extract wavelength information
        self._skip_until(f, '65538')
        self._skip_until(f, '65538')
        o = self._read_floats(f, 8)
        data['centerWavelength'] = o[3]
        data['grating'] = round(o[6])

        self._skip_until(f, '65539')
        self._skip_until_char(f, '.')
        self._back_one_line(f)

        o = self._read_floats(f, 4)
        data['minWavelength'] = o[0]
        data['stepWavelength'] = o[1]
        data['step1Wavelength'] = o[2]
        data['step2Wavelength'] = o[3]
        data['maxWavelength'] = (data['minWavelength'] + 
                                 data['detectorSize'][0] * data['stepWavelength'])

        # Create wavelength axis
        da = np.arange(1, data['detectorSize'][0] + 1)
        data['axisWavelength'] = (
            data['minWavelength'] +
            da * (data['stepWavelength'] + da * data['step1Wavelength'] + da ** 2 * data['step2Wavelength'])
        )

        self._skip_until(f, 'Wavelength')
        self._back_one_line(f)

        data['frameAxis'] = self._read_string(f)
        data['dataType'] = self._read_string(f)
        data['imageAxis'] = self._read_string(f)

        o = self._read_integers(f, 14)
        data['imageArea'] = [[o[0], o[3], o[5]], [o[2], o[1], o[4]]]
        data['frameArea'] = [[o[8], o[11]], [o[10], o[9]]]
        data['frameBins'] = [o[13], o[12]]

        s = (1 + np.diff(data['frameArea'], axis=0)[0]) / data['frameBins']
        z = 1 + np.diff(data['imageArea'][1:])[0]
        data['kineticLength'] = o[4]

        if np.prod(s) != o[7] or np.prod(s) * z != o[6]:
            raise ValueError('Inconsistent image header.')

        self._skip_lines(f, 2 + data['kineticLength'])
        data['imageData'] = np.squeeze(
            np.reshape(
                np.fromfile(f, dtype=np.float32, count=int(np.prod(s) * z)),
                (*s.astype(int), int(z))
            )
        )

        return data

    def _read_string(self, f):
        n = self._read_integers(f, 1)[0]
        if n < 0:
            raise ValueError('Inconsistent string.')
        return f.read(n).decode('utf-8')

    def _read_line(self, f):
        line = f.readline().decode('utf-8')
        if not line:
            raise ValueError('Inconsistent image header.')
        return line.strip()

    def _read_integers(self, f, count):
        return np.fromfile(f, dtype=np.int32, count=count)

    def _read_floats(self, f, count):
        return np.fromfile(f, dtype=np.float32, count=count)

    def _skip_bytes(self, f, count):
        f.seek(count, 1)

    def _skip_lines(self, f, count):
        for _ in range(count):
            self._read_line(f)

    def _skip_until(self, f, string):
        while True:
            line = self._read_line(f)
            if line.startswith(string):
                break

    def _skip_until_char(self, f, char):
        while True:
            c = f.read(1).decode('utf-8')
            if c == char:
                break

    def _back_one_line(self, f):
        while True:
            f.seek(-2, 1)
            c = f.read(1).decode('utf-8')
            if c == '\n':
                break

# Example usage
# reader = SIFReader('example.sif')
# data = reader.read_sif()
