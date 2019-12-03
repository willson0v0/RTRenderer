#include "RTRendererCUDAQT.h"
#include "consts.h"
//#include "misc.h"

RTRendererCUDAQT::RTRendererCUDAQT(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setGeometry(90, 90, MAX_X + 200, MAX_Y + 200);
	pb = new QPushButton("Modify", this);
	pb->setGeometry(10, 10, 50, 50);
	stop_pb = new QPushButton("stop", this);
	stop_pb->setGeometry(60, 10, 50, 50);
	lab = new QLabel(this);
	lab->setGeometry(100, 100 , MAX_X, MAX_Y);
	looper = new LoopThread(this);
	connect(pb, SIGNAL(clicked()), this, SLOT(SlotTest()));
	connect(looper, SIGNAL(refresh_flag()), this, SLOT(refresh()));
	connect(stop_pb, SIGNAL(clicked()), this, SLOT(Stop()));
}

void RTRendererCUDAQT::SlotTest()
{
	looper->start();
}

void LoopThread::run()
{
	kernel();
}


void RTRendererCUDAQT::Stop()
{
	this->looper->break_flag = 1;
}
