#pragma once

int ffmpegToZimgFormat(zimg_image_format& out, const FFMS_Frame* in){
    zimg_image_format_default(&out, ZIMG_API_VERSION);

    out.width = in->EncodedWidth;
    out.height = in->EncodedHeight;

    //default values
    out.color_family = ZIMG_COLOR_YUV;
    out.field_parity = ZIMG_FIELD_PROGRESSIVE;
    out.matrix_coefficients = (out.height > 650) ? ZIMG_MATRIX_BT709 : ZIMG_MATRIX_BT470_BG;
    out.transfer_characteristics = ZIMG_TRANSFER_BT709;
    out.color_primaries = ZIMG_PRIMARIES_BT709;
    out.pixel_range = ZIMG_RANGE_LIMITED;
    switch ((AVPixelFormat)in->EncodedPixelFormat){
        case AV_PIX_FMT_YUVA420P:
        case AV_PIX_FMT_YUV420P:
            out.depth = 8;
            out.subsample_w = 1;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUYV422:
        case AV_PIX_FMT_YUVA422P:
        case AV_PIX_FMT_YUV422P:
            out.depth = 8;
            out.subsample_w = 1;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVA444P:
        case AV_PIX_FMT_YUV444P:
            out.depth = 8;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUV410P:
            out.depth = 8;
            out.subsample_w = 2;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUV411P:
            out.depth = 8;
            out.subsample_w = 2;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUV440P:
            out.depth = 8;
            out.subsample_w = 0;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUVA420P16LE:
        case AV_PIX_FMT_YUV420P16LE:
            out.depth = 16;
            out.subsample_w = 1;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUVA422P16LE:
        case AV_PIX_FMT_YUV422P16LE:
            out.depth = 16;
            out.subsample_w = 1;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVA444P16LE:
        case AV_PIX_FMT_YUV444P16LE:
            out.depth = 16;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVA420P9LE:
        case AV_PIX_FMT_YUV420P9LE:
            out.depth = 9;
            out.subsample_w = 1;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUVA420P10LE:
        case AV_PIX_FMT_YUV420P10LE:
            out.depth = 10;
            out.subsample_w = 1;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUVA422P10LE:
        case AV_PIX_FMT_YUV422P10LE:
            out.depth = 10;
            out.subsample_w = 1;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVA444P9LE:
        case AV_PIX_FMT_YUV444P9LE:
            out.depth = 9;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVA444P10LE:
        case AV_PIX_FMT_YUV444P10LE:
            out.depth = 10;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVA422P9LE:
        case AV_PIX_FMT_YUV422P9LE:
            out.depth = 9;
            out.subsample_w = 1;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUV420P12LE:
            out.depth = 12;
            out.subsample_w = 1;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUV420P14LE:
            out.depth = 14;
            out.subsample_w = 1;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUVA422P12LE:
        case AV_PIX_FMT_YUV422P12LE:
            out.depth = 12;
            out.subsample_w = 1;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUV422P14LE:
            out.depth = 14;
            out.subsample_w = 1;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVA444P12LE:
        case AV_PIX_FMT_YUV444P12LE:
            out.depth = 12;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUV444P14LE:
            out.depth = 14;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUV440P10LE:
            out.depth = 10;
            out.subsample_w = 0;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUV440P12LE:
            out.depth = 12;
            out.subsample_w = 0;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUVJ420P:
            out.pixel_range = ZIMG_RANGE_FULL;
            out.depth = 8;
            out.subsample_w = 1;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_YUVJ422P:
            out.pixel_range = ZIMG_RANGE_FULL;
            out.depth = 8;
            out.subsample_w = 1;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVJ444P:
            out.pixel_range = ZIMG_RANGE_FULL;
            out.depth = 8;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVJ411P:
            out.pixel_range = ZIMG_RANGE_FULL;
            out.depth = 8;
            out.subsample_w = 2;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_YUVJ440P:
            out.pixel_range = ZIMG_RANGE_FULL;
            out.depth = 8;
            out.subsample_w = 0;
            out.subsample_h = 1;
        break;
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_RGBA:
        case AV_PIX_FMT_ARGB:
        case AV_PIX_FMT_ABGR:
        case AV_PIX_FMT_BGRA:
            out.color_family = ZIMG_COLOR_RGB;
            out.matrix_coefficients = ZIMG_MATRIX_RGB;
            out.depth = 8;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        case AV_PIX_FMT_RGB48LE:
        case AV_PIX_FMT_RGBA64LE:
            out.color_family = ZIMG_COLOR_RGB;
            out.matrix_coefficients = ZIMG_MATRIX_RGB;
            out.depth = 16;
            out.subsample_w = 0;
            out.subsample_h = 0;
        break;
        default:
            std::cout << "Unhandled LibAV Pixel Format " << (AVPixelFormat)in->EncodedPixelFormat << std::endl;
            return 1;
    }

    if (out.width%(1 << out.subsample_w) != 0 || out.height%(1 << out.subsample_h)){
        std::cout << "Width and height are not compatible with the subsampling. (For example odd width in YUV4:2:0). This is not supported by zimg" << std::endl;
        return 1;
    }

    if (out.depth <= 8) {
        out.pixel_type = ZIMG_PIXEL_BYTE;
    } else if (out.depth <= 16){
        out.pixel_type = ZIMG_PIXEL_WORD;
    } else {
        std::cout << "unsupported pixel depth" << std::endl;
        return 1;
    }

    if (in->InterlacedFrame){
        if (in->TopFieldFirst){
            out.field_parity = ZIMG_FIELD_TOP;
        } else {
            out.field_parity = ZIMG_FIELD_BOTTOM;
        }
    } else {
        out.field_parity = ZIMG_FIELD_PROGRESSIVE;
    }

    switch ((FFMS_ChromaLocations)in->ChromaLocation){
        case FFMS_LOC_UNSPECIFIED:
            //std::cout << "unspecifed chroma location, defaulting on left" << std::endl;
        case FFMS_LOC_LEFT:
            out.chroma_location = ZIMG_CHROMA_LEFT;
            break;
        case FFMS_LOC_CENTER:
            out.chroma_location = ZIMG_CHROMA_CENTER;
            break;
        case FFMS_LOC_TOPLEFT:
            out.chroma_location = ZIMG_CHROMA_TOP_LEFT;
            break;
        case FFMS_LOC_TOP:
            out.chroma_location = ZIMG_CHROMA_TOP;
            break;
        case FFMS_LOC_BOTTOMLEFT:
            out.chroma_location = ZIMG_CHROMA_BOTTOM_LEFT;
            break;
        case FFMS_LOC_BOTTOM:
            out.chroma_location = ZIMG_CHROMA_BOTTOM;
            break;

        default:
            std::cout << "Unhandled LibAV Chroma position" << std::endl;
            return 1;
    }
    
    switch ((AVColorSpace)in->ColorSpace){
        case AVCOL_SPC_RGB:        
            out.matrix_coefficients = ZIMG_MATRIX_RGB; 
            break;
        case AVCOL_SPC_BT709:    
            out.matrix_coefficients = ZIMG_MATRIX_BT709;                   
            break;
        case AVCOL_SPC_UNSPECIFIED:
            //std::cout << "missing YUV matrix color, guessing..." << std::endl;                 
            break;
        case AVCOL_SPC_FCC:    
            out.matrix_coefficients = ZIMG_MATRIX_FCC;                     
            break;
        case AVCOL_SPC_BT470BG:    
            out.matrix_coefficients = ZIMG_MATRIX_BT470_BG;
            //new default
            out.transfer_characteristics = ZIMG_TRANSFER_BT470_BG;
            out.color_primaries = ZIMG_PRIMARIES_BT470_BG;              
            break;
        case AVCOL_SPC_SMPTE170M:    
            out.matrix_coefficients = ZIMG_MATRIX_ST170_M;                 
            break;
        case AVCOL_SPC_SMPTE240M:    
            out.matrix_coefficients = ZIMG_MATRIX_ST240_M;                 
            break;
        case AVCOL_SPC_YCGCO:    
            out.matrix_coefficients = ZIMG_MATRIX_YCGCO;                   
            break;
        case AVCOL_SPC_BT2020_NCL:    
            out.matrix_coefficients = ZIMG_MATRIX_BT2020_NCL;              
            break;
        case AVCOL_SPC_BT2020_CL:    
            out.matrix_coefficients = ZIMG_MATRIX_BT2020_CL;               
            break;
        //case AVCOL_SPC_SMPTE2085:    
        //    out.matrix_coefficients = ZIMG_MATRIX_ST2085_YDZDX;            
        //    break;
        case AVCOL_SPC_CHROMA_DERIVED_NCL:    
            out.matrix_coefficients = ZIMG_MATRIX_CHROMATICITY_DERIVED_NCL;
            break;
        case AVCOL_SPC_CHROMA_DERIVED_CL:    
            out.matrix_coefficients = ZIMG_MATRIX_CHROMATICITY_DERIVED_CL; 
            break;
        //case AVCOL_SPC_ICTCP:    
        //    out.matrix_coefficients = ZIMG_MATRIX_BT2100_ICTCP;            
        //    break;

        default:
            std::cout << "Unhandled LibAV YUV color matrix" << std::endl;
            return 1;
    }
    
    switch (in->TransferCharateristics){
        case AVCOL_TRC_UNSPECIFIED:
            //std::cout << "missing transfer function, using BT709" << std::endl;;
            break;    
        case AVCOL_TRC_BT709:
            out.transfer_characteristics = ZIMG_TRANSFER_BT709;
            break;
        case AVCOL_TRC_GAMMA22:
            out.transfer_characteristics = ZIMG_TRANSFER_BT470_M;
            break;
        case AVCOL_TRC_GAMMA28:
            out.transfer_characteristics = ZIMG_TRANSFER_BT470_BG;
            break;
        case AVCOL_TRC_SMPTE170M:
            out.transfer_characteristics = ZIMG_TRANSFER_BT601;
            break;
        case AVCOL_TRC_SMPTE240M:
            out.transfer_characteristics = ZIMG_TRANSFER_ST240_M;
            break;
        case AVCOL_TRC_LINEAR:
            out.transfer_characteristics = ZIMG_TRANSFER_LINEAR;
            break;
        case AVCOL_TRC_LOG:
            out.transfer_characteristics = ZIMG_TRANSFER_LOG_100;
            break;
        case AVCOL_TRC_LOG_SQRT:
            out.transfer_characteristics = ZIMG_TRANSFER_LOG_316;
            break;
        case AVCOL_TRC_IEC61966_2_4:
            out.transfer_characteristics = ZIMG_TRANSFER_IEC_61966_2_4;
            break;
        //case AVCOL_TRC_BT1361_ECG:
        //    out.transfer_characteristics = ZIMG_TRANSFER_BT1361;
        //    break;
        case AVCOL_TRC_IEC61966_2_1:
            out.transfer_characteristics = ZIMG_TRANSFER_IEC_61966_2_1;
            break;
        case AVCOL_TRC_BT2020_10:
            out.transfer_characteristics = ZIMG_TRANSFER_BT2020_10;
            break;
        case AVCOL_TRC_BT2020_12:
            out.transfer_characteristics = ZIMG_TRANSFER_BT2020_12;
            break;
        case AVCOL_TRC_SMPTE2084:
            out.transfer_characteristics = ZIMG_TRANSFER_ST2084;
            break;
        case AVCOL_TRC_SMPTE428:
            out.transfer_characteristics = ZIMG_TRANSFER_ST428;
            break;
        case AVCOL_TRC_ARIB_STD_B67:
            out.transfer_characteristics = ZIMG_TRANSFER_ARIB_B67;
            break;

        default:
            std::cout << "Unhandled LibAV color transfer function" << std::endl;
            return 1;
    }
    
    switch (in->ColorPrimaries){
        case AVCOL_PRI_UNSPECIFIED:
            //std::cout << "unspecified primaries, defaulting to BT709" << std::endl;
            break;
        case AVCOL_PRI_BT709:
            out.color_primaries = ZIMG_PRIMARIES_BT709;
            break;
        case AVCOL_PRI_BT470M:
            out.color_primaries = ZIMG_PRIMARIES_BT470_M;
            break;
        case AVCOL_PRI_BT470BG:
            out.color_primaries = ZIMG_PRIMARIES_BT470_BG;
            break;
        case AVCOL_PRI_SMPTE170M:
            out.color_primaries = ZIMG_PRIMARIES_ST170_M;
            break;
        case AVCOL_PRI_SMPTE240M:
            out.color_primaries = ZIMG_PRIMARIES_ST240_M;
            break;
        case AVCOL_PRI_FILM:
            out.color_primaries = ZIMG_PRIMARIES_FILM;
            break;
        case AVCOL_PRI_BT2020:
            out.color_primaries = ZIMG_PRIMARIES_BT2020;
            break;
        case AVCOL_PRI_SMPTE428:
            out.color_primaries = ZIMG_PRIMARIES_ST428;
            break;
        case AVCOL_PRI_SMPTE431:
            out.color_primaries = ZIMG_PRIMARIES_ST431_2;
            break;
        case AVCOL_PRI_SMPTE432:
            out.color_primaries = ZIMG_PRIMARIES_ST432_1;
            break;
        case AVCOL_PRI_EBU3213:
            out.color_primaries = ZIMG_PRIMARIES_EBU3213_E;
            break;
            
        default:
            std::cout << "Unhandled LibAV color primaries" << std::endl;
            return 1;
    }

    switch (in->ColorRange){
        case AVCOL_RANGE_UNSPECIFIED:
            //std::cout << "Warning: unspecified color range, defaulting to full" << std::endl;
            break;
        case AVCOL_RANGE_MPEG:
            out.pixel_range = ZIMG_RANGE_LIMITED;
            break;
        case AVCOL_RANGE_JPEG:
            out.pixel_range = ZIMG_RANGE_FULL;
            break;
        default:
            std::cout << "Unhandled LibAV color range object received " << std::endl;
            return 1;
    }
    return 0;
}