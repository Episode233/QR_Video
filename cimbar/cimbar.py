#!/usr/bin/python3

"""color-icon-matrix barcode

Usage:
  ./cimbar.py <IMAGES>... --output=<filename> [--config=<sq8x8,sq5x5,sq5x6>] [--dark | --light]
                         [--colorbits=<0-3>] [--deskew=<0-2>] [--ecc=<0-200>]
                         [--fountain] [--preprocess=<0,1>] [--color-correct=<0-2>]
  ./cimbar.py --encode (<src_data> | --src_data=<filename>) (<output> | --output=<filename>)
                       [--config=<sq8x8,og8x8,sq5x5,sq5x6>] [--dark | --light]
                       [--colorbits=<0-3>] [--ecc=<0-150>] [--fountain]
  ./cimbar.py (-h | --help)

Examples:
  python -m cimbar --encode myfile.txt cimb-code.png
  python -m cimbar cimb-code.png -o myfile.txt

Options:
  -h --help                        Show this help.
  --version                        Show version.
  --src_data=<filename>            For encoding. Data to encode.
  -o --output=<filename>           For encoding. Where to store output. For encodes, this may be interpreted as a prefix.
  -c --colorbits=<0-3>             How many colorbits in the image. [default: 2]
  -e --ecc=<0-200>                 Reed solomon error correction level. 0 is no ecc. [default: auto]
  -f --fountain                    Use fountain encoding scheme.
  --config=<config>                Choose configuration from sq8x8,sq5x5,sq5x6. [default: sq8x8]
  --dark                           Use dark palette. [default]
  --light                          Use light palette.
  --color-correct=<0-7>            Color correction. 0 is off. 1 is white balance. 3 is 2-pass on a fountain-encoded image. [default: 1]
  --deskew=<0-2>                   Deskew level. 0 is no deskew. Should usually be 0 or default. [default: 1]
  --preprocess=<0,1>               Sharpen image before decoding. Default is to guess. [default: -1]
"""
import os
from collections import defaultdict
from io import BytesIO
from os import path
from tempfile import TemporaryDirectory

import cv2
import numpy
from docopt import docopt
from PIL import Image
from reedsolo import RSCodec

from cimbar import conf
from cimbar.deskew.deskewer import deskewer
from cimbar.encode.cell_positions import cell_positions, AdjacentCellFinder, FloodDecodeOrder
from cimbar.encode.cimb_translator import CimbEncoder, CimbDecoder, avg_color, possible_colors
from cimbar.encode.rss import reed_solomon_stream
from cimbar.fountain.header import fountain_header
from cimbar.util.bit_file import bit_file
from cimbar.util.interleave import interleave, interleave_reverse, interleaved_writer


BITS_PER_COLOR=conf.BITS_PER_COLOR


class BlockEncoderStream:
    """用于大文件分块编码的类"""

    def __init__(self, f, block_size, ecc):
        self.f = f
        self.block_size = block_size
        self.ecc = ecc

        # 从文件名获取文件类型(扩展名),限制为 4 字符
        filename = getattr(f, 'name', '')
        ext = filename.split('.')[-1] if '.' in filename else ''
        self.file_type = ext.zfill(4).encode('ascii')  # 添加 0 前缀并补齐到 4 字符

        self.file_size = f.seek(0, 2) + len(self.file_type)  # 获取文件大小
        f.seek(0)  # 重置文件指针
        self.current_pos = 0
        self._closed = False
        self._type_written = False  # 标记是否已写入文件类型
        self._index_written = False  # 标记是否已写入索引

        # 计算RS参数
        self.rs_block_size = conf.ECC_BLOCK_SIZE
        self.data_block_size = self.rs_block_size - ecc

        # 计算每帧实际可以存储的数据大小
        num_rs_blocks = block_size // self.rs_block_size
        self.frame_data_size = num_rs_blocks * self.data_block_size - 8  # 减去索引大小

        # 计算总共需要的帧数（向上取整）
        self.total_frames = (self.file_size + self.frame_data_size - 1) // self.frame_data_size
        self.current_frame = 0

        # 初始化Reed-Solomon编码器列表
        self.rsc_list = []
        if ecc:
            for _ in range(num_rs_blocks):
                self.rsc_list.append(RSCodec(ecc, nsize=self.rs_block_size, fcr=1, prim=0x187))

    def read(self, size):
        if self._closed:
            raise ValueError("I/O operation on closed file")

        if self.current_frame >= self.total_frames:
            return b''

        index = str(self.current_frame).zfill(8).encode('ascii')
        block = index

        # 首先写入文件类型标识
        if not self._type_written:
            self._type_written = True
            block += self.file_type
            remaining_size = self.file_size - len(block)
            data_to_read = min(self.frame_data_size - len(self.file_type), remaining_size)
            block += self.f.read(data_to_read)
        else:
            # 读取这一帧需要的实际数据量
            remaining_size = self.file_size - self.current_pos
            data_to_read = min(self.frame_data_size, remaining_size)
            block += self.f.read(data_to_read)
        self.current_pos += data_to_read
        self.current_frame += 1

        if not self.rsc_list:
            # 如果没有启用ECC，填充到块大小
            if len(block) < self.block_size:
                block = block + b'\0' * (self.block_size - len(block))
            return block

        # 对数据进行RS编码
        encoded_data = b''
        offset = 0
        for rsc in self.rsc_list:
            if offset >= len(block):
                # 如果没有更多数据，使用空数据块进行编码
                curr_data = b'\0' * self.data_block_size
            else:
                # 获取当前RS块的数据
                curr_data = block[offset:offset + self.data_block_size]
                if len(curr_data) < self.data_block_size:
                    curr_data = curr_data + b'\0' * (self.data_block_size - len(curr_data))

            # 进行RS编码
            encoded = rsc.encode(curr_data)
            encoded_data += encoded
            offset += self.data_block_size

        return encoded_data

    @property
    def closed(self):
        return self._closed

    def close(self):
        if not self._closed:
            self.f.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class BlockDecoderStream:
    """用于大文件分块解码的类"""

    def __init__(self, f, block_size, ecc):
        self.f = f
        self.block_size = block_size
        self.ecc = ecc
        self._closed = False
        self._write_time = 0
        self._content = None


        # 使用与编码器相同的参数
        self.rs_block_size = conf.ECC_BLOCK_SIZE
        self.data_block_size = self.rs_block_size - ecc

        # 计算每帧的实际数据大小
        num_rs_blocks = block_size // self.rs_block_size
        self.frame_data_size = num_rs_blocks * self.data_block_size

        # 初始化Reed-Solomon解码器列表
        self.rsc_list = []
        if ecc:
            for _ in range(num_rs_blocks):
                self.rsc_list.append(RSCodec(ecc, nsize=self.rs_block_size, fcr=1, prim=0x187))

        # 记录实际写入的数据大小
        self.total_data_size = None
        self.current_data_size = 0

    def write(self, data):
        self._write_time += 1

        if self._closed:
            raise ValueError("I/O operation on closed file")

        if not data or len(data) == 0:
            return

        if not self.rsc_list:
            # 如果没有启用ECC，直接写入数据
            valid_data = data.rstrip(b'\0')
            if valid_data and (self.total_data_size is None or
                               self.current_data_size + len(valid_data) <= self.total_data_size):
                self.f.write(valid_data)
                self.current_data_size += len(valid_data)
            return

        try:
            # 分块进行RS解码
            decoded_data = b''
            offset = 0
            for rsc in self.rsc_list:
                if offset >= len(data):
                    break

                # 获取当前RS块
                curr_block = data[offset:offset + self.rs_block_size]
                if len(curr_block) == 0:
                    break

                # 确保块大小正确
                if len(curr_block) < self.rs_block_size:
                    curr_block = curr_block + b'\0' * (self.rs_block_size - len(curr_block))
                try:
                    # 解码当前块
                    decoded = bytes(rsc.decode(curr_block)[0])
                    decoded_data += decoded
                    offset += self.rs_block_size
                except Exception as e:
                    print(f"Warning: RS decode error at offset {offset}: {e}")
                    break

            # 如果已知总大小，只写入需要的数据
            if decoded_data:
                if self.total_data_size is not None:
                    # 计算还需要写入多少数据
                    remaining = self.total_data_size - self.current_data_size
                    if remaining > 0:
                        # 如果还需要写入数据，去掉末尾的零
                        to_write = decoded_data[:remaining].rstrip(b'\0')
                        if to_write:
                            self.f.write(to_write)
                            self.current_data_size += len(to_write)
                else:
                    # 如果不知道总大小，写入所有非零数据
                    to_write = decoded_data.rstrip(b'\0')
                    if self._write_time == 1:
                        self._content = to_write
                    if self._write_time == 2:
                        self._content += to_write
                        self._content = self._content.decode('utf-8')
                        print(self._content)
                    if to_write:
                        self.f.write(to_write)
                        self.current_data_size += len(to_write)

        except Exception as e:
            print(f"Error in block decoding: {e}")

    @property
    def closed(self):
        return self._closed

    def close(self):
        if not self._closed:
            self.f.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def get_deskew_params(level):
    level = int(level)
    return {
        'deskew': level,
        'auto_dewarp': level >= 2,
    }


def bits_per_op():
    return conf.BITS_PER_SYMBOL + BITS_PER_COLOR


def use_split_mode():
    return getattr(conf, 'SPLIT_MODE', True)


def num_cells():
    return conf.CELL_DIM_Y*conf.CELL_DIM_X - (conf.MARKER_SIZE_X*conf.MARKER_SIZE_Y * 4)


def num_fountain_blocks():
    return bits_per_op() * 2


def capacity(bits_per_op=bits_per_op()):
    return num_cells() * bits_per_op // 8;


def _fountain_chunk_size(ecc=conf.ECC, bits_per_op=bits_per_op(), fountain_blocks=conf.FOUNTAIN_BLOCKS):
    fountain_blocks = fountain_blocks or num_fountain_blocks()
    return capacity(bits_per_op) * (conf.ECC_BLOCK_SIZE-ecc) // conf.ECC_BLOCK_SIZE // fountain_blocks


def _get_expected_fountain_headers(headers, bits_per_symbol=conf.BITS_PER_SYMBOL, bits_per_color=BITS_PER_COLOR):
    import bitstring
    from bitstring import Bits, BitStream

    # it'd be nice to use the frame id as well, but sometimes we skip frames.
    # specifically, at the end of the input data (so when num_cunks*chunk size ~= file size)
    # we will usually skip a frame (whenever the last chunk is not conveniently equal to the frame size)
    # we *could* do that math and use the frame id anyway, it might be worth it...
    for header in headers:
        if not header.bad():
            break
    assert not header.bad()  # TODO: maybe just return NULL?

    color_headers = []
    for _ in range(bits_per_color * 2):
        color_headers += bytes(header)[:-2]  # remove frame id

    print(color_headers)

    res = []
    stream = BitStream()
    stream.append(Bits(bytes=color_headers))
    while stream.pos < stream.length:
        res.append(stream.read(f'uint:{bits_per_color}'))
    return res


def _get_fountain_header_cell_index(cells, expected_vals):
    # TODO: misleading to say this works for all FOUNTAIN_BLOCKS values...
    fountain_blocks = conf.FOUNTAIN_BLOCKS or num_fountain_blocks()
    end = capacity(BITS_PER_COLOR) * 8 // BITS_PER_COLOR
    header_start_interval = capacity(bits_per_op()) * 8 // fountain_blocks // BITS_PER_COLOR
    header_len = (fountain_header.length-2) * 8 // BITS_PER_COLOR

    cell_idx = []
    i = 0
    while i < end:
        # maybe split this into a list of lists? idk
        cell_idx += list(range(i, i+header_len))
        i += header_start_interval

    # sanity check, we're doomed if this fails
    assert len(cell_idx) == len(expected_vals), f'{len(cell_idx)} == {len(expected_vals)}'
    res = defaultdict(list)
    for idx,exp in zip(cell_idx, expected_vals):
        res[exp].append(cells[idx])
    return res


def _build_color_decode_lookups(ct, color_img, color_map):
    res = defaultdict(list)
    for exp, pos_list in color_map.items():
        for pos in pos_list:
            cell = _crop_cell(color_img, pos[0], pos[1])
            color = avg_color(cell, dark=ct.dark)
            res[exp].append(color)
            bits = ct.decode_color(cell, 0)
            if bits != exp:
                print(f' wrong!!! {pos} ... {bits} == {exp}')

    # return averages
    return {
        k: tuple(numpy.mean(vals, axis=0)) for k,vals in res.items()
    }


def _decode_sector_calc(midpt, x, y, num_sectors):
    if num_sectors < 2:
        return 0
    if (x - midpt[0])**2 + (y - midpt[1])**2 < 400**2:
        return 0
    else:
        return 1


def _derive_color_lookups(ct, color_img, cells, fount_headers, splits=0):
    header_cell_locs = _get_fountain_header_cell_index(
        list(interleave(cells, conf.INTERLEAVE_BLOCKS, conf.INTERLEAVE_PARTITIONS)),
        _get_expected_fountain_headers(fount_headers),
    )
    print(header_cell_locs)

    color_maps = []
    if splits == 2:
        center_map = defaultdict(list)
        edge_map = defaultdict(list)
        midX = conf.TOTAL_SIZE // 2
        midY = conf.TOTAL_SIZE // 2
        for exp,pos in header_cell_locs.items():
            for xy in pos:
                if _decode_sector_calc((midX, midY), *xy, splits) == 0:
                    center_map[exp].append(xy)
                else:
                    edge_map[exp].append(xy)

        lc = {exp: len(pos) for exp, pos in center_map.items()}
        le = {exp: len(pos) for exp, pos in edge_map.items()}
        print(f'sanity check. len(center)={lc}, len(edge)={le}')
        color_maps = [center_map, edge_map]

    else:
        color_map = dict()
        for exp,pos in header_cell_locs.items():
            color_map[exp] = pos
        color_maps = [color_map]

    return [_build_color_decode_lookups(ct, color_img, cm) for cm in color_maps]


def detect_and_deskew(src_image, temp_image, dark, auto_dewarp=False):
    return deskewer(src_image, temp_image, dark, auto_dewarp=auto_dewarp)


def _decode_cell(ct, img, x, y, drift):
    best_distance = 1000
    for dx, dy in drift.pairs:
        testX = x + drift.x + dx
        testY = y + drift.y + dy
        img_cell = img.crop((testX, testY, testX + conf.CELL_SIZE, testY + conf.CELL_SIZE))
        bits, min_distance = ct.decode_symbol(img_cell)
        best_distance = min(min_distance, best_distance)
        if min_distance == best_distance:
            best_bits = bits
            best_dx = dx
            best_dy = dy
        if min_distance < 8:
            break

    testX = x + drift.x + best_dx
    testY = y + drift.y + best_dy
    best_cell = (testX, testY)
    return best_bits, best_cell, best_dx, best_dy, best_distance


def _preprocess_for_decode(img):
    ''' This might need to be conditional based on source image size.'''
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    kernel = numpy.array([[-1.0,-1.0,-1.0], [-1.0,8.5,-1.0], [-1.0,-1.0,-1.0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    return img


def _get_decoder_stream(outfile, ecc, fountain):
    # set up the outstream: image -> reedsolomon -> fountain -> zstd_decompress -> raw bytes
    f = open(outfile, 'wb')
    if fountain:
        import zstandard as zstd
        from cimbar.fountain.fountain_decoder_stream import fountain_decoder_stream
        decompressor = zstd.ZstdDecompressor().stream_writer(f)
        f = fountain_decoder_stream(decompressor, _fountain_chunk_size(ecc))
    # on_rss_failure = b'' if fountain else None
    # stream = reed_solomon_stream(f, ecc, conf.ECC_BLOCK_SIZE, mode='write', on_failure=on_rss_failure) if ecc else f
    # fount = f if fountain else None
    # return stream, fount
    else:
        # 新的分块处理逻辑
        block_size = capacity()
        f = BlockDecoderStream(f, block_size, ecc)

    return f, None


def compute_tint(img, dark):
    def update(c, r, g, b):
        c['r'] = max(c['r'], r)
        c['g'] = max(c['g'], g)
        c['b'] = max(c['b'], b)

    cc = {}
    cc['r'] = cc['g'] = cc['b'] = 1

    if dark:
        pos = [(28, 28), (28, conf.TOTAL_SIZE-32), (conf.TOTAL_SIZE-32, 28)]
    else:
        pos = [(67, 0), (0, 67), (conf.TOTAL_SIZE-79, 0), (0, conf.TOTAL_SIZE-79)]

    colors = []
    for x, y in pos:
        iblock = img.crop((x, y, x + 4, y + 4))
        ac = avg_color(iblock, False)
        colors.append(ac)
        update(cc, *ac)

    print(f'tint is {cc}')
    return cc['r'], cc['g'], cc['b']


def _crop_cell(img, x, y):
    return img.crop((x+1, y+1, x + conf.CELL_SIZE-1, y + conf.CELL_SIZE-1))


def _decode_symbols(ct, img):
    cell_pos, num_edge_cells = cell_positions(conf.CELL_SPACING_X, conf.CELL_SPACING_Y, conf.CELL_DIM_X,
                                              conf.CELL_DIM_Y, conf.CELLS_OFFSET, conf.MARKER_SIZE_X, conf.MARKER_SIZE_Y)
    finder = AdjacentCellFinder(cell_pos, num_edge_cells, conf.CELL_DIM_X, conf.MARKER_SIZE_X)
    decode_order = FloodDecodeOrder(cell_pos, finder)
    print('beginning decode symbols pass...')
    for i, (x, y), drift in decode_order:
        best_bits, best_cell, best_dx, best_dy, best_distance = _decode_cell(ct, img, x, y, drift)
        decode_order.update(best_dx, best_dy, best_distance)
        yield i, best_bits, best_cell


def _calc_ccm(ct, color_lookups, cc_setting, state_info):
    splits = 2 if cc_setting in (6, 7) else 0
    if cc_setting in (3, 4, 5):
        possible = possible_colors(ct.dark, BITS_PER_COLOR)
        if len(color_lookups[0]) < len(possible):
            raise Exception("kaboomski")  # not clear whether this should throw or not, really.
        exp = [color for i,color in enumerate(possible) if i in color_lookups[0]] + [(255,255,255)]
        exp = numpy.array(exp)
        white = state_info['white']
        observed = numpy.array([v for k,v in sorted(color_lookups[0].items())] + [white])
        from colour.characterisation.correction import matrix_colour_correction_Cheung2004
        der = matrix_colour_correction_Cheung2004(observed, exp)

        # not sure which of this would be better...
        if ct.ccm is None or cc_setting == 4:
            ct.ccm = der
        else:  # cc_setting == 3,5
            ct.ccm = der.dot(ct.ccm)

    if splits:  # 6,7
        from colour.characterisation.correction import matrix_colour_correction_Cheung2004
        exp = numpy.array(possible_colors(ct.dark, BITS_PER_COLOR) + [(255,255,255)])
        white = state_info['white']
        ccms = list()
        i = 0
        while i < splits:
            observed = numpy.array([v for k,v in sorted(color_lookups[i].items())] + [white])
            der = matrix_colour_correction_Cheung2004(observed, exp)
            ccms.append(der)
            i += 1

        if ct.ccm is None or cc_setting == 7:
            ct.ccm = ccms
        else:
            ct.ccm = [der.dot(ct.ccm) for der in ccms]

    if cc_setting == 5:
        ct.colors = color_lookups[0]
    if cc_setting == 10:
        ct.disable_color_scaling = True
        ct.colors = color_lookups[0]


def _decode_iter(ct, img, color_img, state_info={}):
    decoding = sorted(_decode_symbols(ct, img))
    if use_split_mode():
        for i, bits, _ in decoding:
            yield i, bits
        yield -1, None
    # state_info can be set at any time, but it will probably be set by the caller *after* the empty yield above
    if state_info.get('color_correct') == 1:
        white = state_info['white']
        from colormath.chromatic_adaptation import _get_adaptation_matrix
        ct.ccm = _get_adaptation_matrix(numpy.array([*white]),
                                        numpy.array([255, 255, 255]), 2, 'von_kries')

    if state_info.get('headers'):
        cc_setting = state_info['color_correct']
        splits = 2 if cc_setting in (6, 7) else 0

        cells = [cell for _, __, cell in decoding]
        color_lookups = _derive_color_lookups(ct, color_img, cells, state_info.get('headers'), splits)
        print('color lookups:')
        print(color_lookups)

        _calc_ccm(ct, color_lookups, cc_setting, state_info)
    print('beginning decode colors pass...')
    midX = conf.TOTAL_SIZE // 2
    midY = conf.TOTAL_SIZE // 2
    for i, bits, cell in decoding:
        testX, testY = cell
        best_cell = _crop_cell(color_img, testX, testY)
        decode_sector = 0 if ct.ccm is None else _decode_sector_calc((midX, midY), testX, testY, len(ct.ccm))
        if use_split_mode():
            yield i, ct.decode_color(best_cell, 0)
        else:
            yield i, bits + (ct.decode_color(best_cell, 0) << conf.BITS_PER_SYMBOL)


def decode_iter(src_image, dark, should_preprocess, color_correct, deskew, auto_dewarp, state_info={}):
    tempdir = None
    if deskew:
        tempdir = TemporaryDirectory()
        temp_img = path.join(tempdir.name, path.basename(src_image))  # or /tmp
        dims = detect_and_deskew(src_image, temp_img, dark, auto_dewarp)
        if should_preprocess < 0:
            should_preprocess = dims[0] < conf.TOTAL_SIZE or dims[1] < conf.TOTAL_SIZE
        color_img = Image.open(temp_img)
    else:
        color_img = Image.open(src_image)

    ct = CimbDecoder(dark, symbol_bits=conf.BITS_PER_SYMBOL, color_bits=conf.BITS_PER_COLOR)
    img = _preprocess_for_decode(color_img) if should_preprocess else color_img

    if color_correct:
        white = compute_tint(color_img, dark)
        state_info['white'] = white
        state_info['color_correct'] = color_correct
    yield from _decode_iter(ct, img, color_img, state_info)
    if tempdir:  # cleanup
        with tempdir:
            pass


def decode(src_images, outfile, dark=True, ecc=conf.ECC, fountain=False, force_preprocess=False, color_correct=False,
           deskew=True, auto_dewarp=False):
    cells, _ = cell_positions(conf.CELL_SPACING_X, conf.CELL_SPACING_Y, conf.CELL_DIM_X, conf.CELL_DIM_Y,
                              conf.CELLS_OFFSET, conf.MARKER_SIZE_X, conf.MARKER_SIZE_Y)
    interleave_lookup, block_size = interleave_reverse(cells, conf.INTERLEAVE_BLOCKS, conf.INTERLEAVE_PARTITIONS)
    dstream, fount = _get_decoder_stream(outfile, ecc, fountain)
    dupe_stream = dupe_pass = None
    if color_correct >= 3 and not fount:
        dupe_stream, fount = _get_decoder_stream('/dev/null', ecc, True)
    with dstream as outstream:
        for imgf in src_images:
            if use_split_mode():
                first_pass = interleaved_writer(
                    f=outstream, bits_per_op=conf.BITS_PER_SYMBOL, mode='write', keep_open=True
                )
                if dupe_stream:
                    dupe_pass = interleaved_writer(
                        f=dupe_stream, bits_per_op=conf.BITS_PER_SYMBOL, mode='write', keep_open=True
                    )
                second_pass = interleaved_writer(
                    f=outstream, bits_per_op=BITS_PER_COLOR, mode='write', keep_open=True
                )
            else:
                first_pass = interleaved_writer(f=outstream, bits_per_op=bits_per_op(), mode='write', keep_open=True)
                second_pass = None

            # this is a bit goofy, might refactor it to have less "loop through writers" weirdness
            iw = first_pass
            state_info = {}
            for i, bits in decode_iter(
                    imgf, dark, force_preprocess, color_correct, deskew, auto_dewarp, state_info
            ):
                if i == -1:
                    # flush and move to the second writer
                    with iw:
                        pass
                    if dupe_pass:
                        with dupe_pass:
                            pass
                    iw = second_pass
                    if fount:
                        state_info['headers'] = fount.headers
                    continue
                block = interleave_lookup[i] // block_size
                iw.write(bits, block)
                if dupe_pass:
                    dupe_pass.write(bits, block)

            # flush iw
            with iw:
                pass


def _get_image_template(width, dark):
    bitmap_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bitmap'))
    color = (0, 0, 0) if dark else (255, 255, 255)
    img = Image.new('RGB', (width, width), color=color)

    suffix = 'dark' if dark else 'light'
    anchor = Image.open(os.path.join(bitmap_path, f"anchor-{suffix}.png"))
    anchor_br = Image.open(os.path.join(bitmap_path, f"anchor-secondary-{suffix}.png"))
    aw, ah = anchor.size
    img.paste(anchor, (0, 0))
    img.paste(anchor, (0, width-ah))
    img.paste(anchor, (width-aw, 0))
    img.paste(anchor_br, (width-aw, width-ah))

    horizontal_guide = Image.open(os.path.join(bitmap_path, f"guide-horizontal-{suffix}.png"))
    gw, _ = horizontal_guide.size
    img.paste(horizontal_guide, (width//2 - gw//2, 2))
    img.paste(horizontal_guide, (width//2 - gw//2, width-4))
    img.paste(horizontal_guide, (width//2 - gw - gw//2, width-4))  # long bottom guide
    img.paste(horizontal_guide, (width//2 + gw - gw//2, width-4))  # ''

    vertical_guide = Image.open(os.path.join(bitmap_path, f"guide-vertical-{suffix}.png"))
    _, gh = vertical_guide.size
    img.paste(vertical_guide, (2, width//2 - gw//2))
    img.paste(vertical_guide, (width-4, width//2 - gw//2))
    return img


def _get_encoder_stream(src, ecc, fountain, compression_level=16):
    # various checks to set up the instream.
    # the hierarchy is raw bytes -> zstd -> fountain -> reedsolomon -> image
    f = open(src, 'rb')
    if fountain:
        import zstandard as zstd
        from cimbar.fountain.fountain_encoder_stream import fountain_encoder_stream
        reader = zstd.ZstdCompressor(level=compression_level).stream_reader(f)
        f = fountain_encoder_stream(reader, _fountain_chunk_size(ecc))
        read_size = _fountain_chunk_size(ecc)
        read_count = (f.len // read_size) * 2
    else:
        # read_size = capacity()
        # file_size = f.seek(0, 2)  # 获取文件大小
        # f.seek(0)  # 重置文件指针
        # read_count = (file_size + read_size - 1) // read_size

        # 新的分块处理逻辑
        block_size = capacity()  # 每个块的大小
        f = BlockEncoderStream(f, block_size, ecc)
    params = {
        'read_size': f.frame_data_size,
        'read_count': f.total_frames,
    }
    print(params)
    return f, params


def encode_iter(src_data, ecc, fountain):
    estream, params = _get_encoder_stream(src_data, ecc, fountain)
    with estream as instream, bit_file(instream, bits_per_op=bits_per_op(), **params) as f:
        frame_num = 0
        while f.read_count > 0:
            cells, _ = cell_positions(conf.CELL_SPACING_X, conf.CELL_SPACING_Y, conf.CELL_DIM_X, conf.CELL_DIM_Y,
                                      conf.CELLS_OFFSET, conf.MARKER_SIZE_X, conf.MARKER_SIZE_Y)
            assert len(cells) == num_cells()

            if use_split_mode():
                symbols = []
                for x, y in interleave(cells, conf.INTERLEAVE_BLOCKS, conf.INTERLEAVE_PARTITIONS):
                    bits = f.read(conf.BITS_PER_SYMBOL)
                    symbols.append(bits)

                # there are better ways to do this than reverse+pop...
                # the important part is that it's a 2-pass approach
                symbols.reverse()

                for x, y in interleave(cells, conf.INTERLEAVE_BLOCKS, conf.INTERLEAVE_PARTITIONS):
                    bits = symbols.pop() | (f.read(BITS_PER_COLOR) << conf.BITS_PER_SYMBOL)
                    yield bits, x, y, frame_num

            else:
                for x, y in interleave(cells, conf.INTERLEAVE_BLOCKS, conf.INTERLEAVE_PARTITIONS):
                    bits = f.read()
                    yield bits, x, y, frame_num

            frame_num += 1
        print(f'encoded {frame_num} frames')


def encode(src_data, dark=True, ecc=conf.ECC, fountain=False):
    img = None
    frame = None
    imagelist = []
    ct = CimbEncoder(dark, symbol_bits=conf.BITS_PER_SYMBOL, color_bits=BITS_PER_COLOR)
    for bits, x, y, frame_num in encode_iter(src_data, ecc, fountain):
        if frame != frame_num:  # save
            if img is not None:
                imagelist.append(img)
            img = _get_image_template(conf.TOTAL_SIZE, dark)
            frame = frame_num
        encoded = ct.encode(bits)
        img.paste(encoded, (x, y))
    imagelist.append(img)
    return imagelist


def main():
    import glob
    args = docopt(__doc__, version='cimbar 0.6.0')

    global BITS_PER_COLOR
    BITS_PER_COLOR = int(args.get('--colorbits'))

    config = args['--config']
    if config:
        config = conf.known[config]
        conf.init(config)
    dark = args['--dark'] or not args['--light']
    try:
        ecc = int(args.get('--ecc'))
    except:
        ecc = conf.ECC
    fountain = bool(args.get('--fountain'))

    if args['--encode']:
        src_data = args['<src_data>'] or args['--src_data']
        dst_image = args['<output>'] or args['--output']
        encode(src_data, dst_image, dark, ecc, fountain)
        return

    deskew = get_deskew_params(args.get('--deskew'))
    should_preprocess = int(args.get('--preprocess'))
    color_correct = int(args.get('--color-correct'))
    # 处理通配符
    src_images = []
    for pattern in args['<IMAGES>']:
        src_images.extend(sorted(glob.glob(pattern)))

    if not src_images:
        print("No input files found!")
        return
    dst_data = args['<output>'] or args['--output']
    decode(src_images, dst_data, dark, ecc, fountain, should_preprocess, color_correct, **deskew)


if __name__ == '__main__':
    main()


