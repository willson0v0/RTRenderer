#include "RTRendererCUDAQT.h"

RTRendererCUDAQT::RTRendererCUDAQT(QWidget *parent)
	: QMainWindow(parent)
{
	launchKernal();
	ui.setupUi(this);
}
