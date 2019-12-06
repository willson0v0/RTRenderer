#include "RTRendererCUDAQT.h"
#include <QImage>
#include <QPainter>
#include "ui_RTRendererCUDAQT.h"
#include <QtWidgets/QApplication>
#include "consts.h"
#include <string>
#include <sstream>



//认真布局了
//改名字 一一对应 Lab->Lab StartButton->StartButton StopButton ParameterText->ParameterText  Para_Stop->Para_Stop
//函数名

RTRendererCUDAQT::RTRendererCUDAQT(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setGeometry(10, 10, MAX_X + 400, MAX_Y + 400);

	Lab = new QLabel(this);
	Lab->setGeometry(20, 20, MAX_X, MAX_Y);

	logText = new QTextEdit(this);
	logText->setGeometry(MAX_X + 50, 20, 300, MAX_Y);

	StartButton = new QPushButton("Start", this);
	StartButton->setGeometry(MAX_X + 50,MAX_Y + 40 ,100 ,20 );

	StopButton = new QPushButton("Stop", this);
	StopButton->setGeometry(MAX_X + 50, MAX_Y + 80 , 100, 20);

	Updater = new QPushButton("Update", this);
	Updater->setGeometry(MAX_X + 50, MAX_Y + 120, 100, 20);

	Discarder = new QPushButton("Discard", this);
	Discarder->setGeometry(MAX_X + 50, MAX_Y + 160, 100, 20);


	
	



	paraTargetSPP = new QLineEdit(this);
	paraTargetSPP->setGeometry(20, MAX_Y + 20, 80, 20);

	labTargetSPP = new QLabel(this);
	labTargetSPP->setGeometry(20, MAX_Y + 40, 80, 20);
	labTargetSPP->setText("TargetSPP");

	

	paraClipUpperbound = new QLineEdit(this);
	paraClipUpperbound->setGeometry(120, MAX_Y + 20, 80, 20);

	labClipUpperbound = new QLabel(this);
	labClipUpperbound->setGeometry(120, MAX_Y + 40, 80, 20);
	labClipUpperbound->setText("ClipUpperbound");



	looper = new LoopThread(this);
	connect(StartButton, SIGNAL(clicked()), this, SLOT(Startear()));
	connect(looper, SIGNAL(refresh_flag()), this, SLOT(refresh()));
	connect(StopButton, SIGNAL(clicked()), this, SLOT(Stop()));
	connect(Updater, SIGNAL(clicked()), this, SLOT(setParameter()));
	connect(Discarder, SIGNAL(clicked()), this, SLOT(discardParameter()));
}



void LoopThread::run()
{
	kernel();
}


//给自己加个模板，还有其他一些东西。
void LoopThread::PrintMessege()
{
	emit info_flag();
}

void LoopThread::checkBreak()
{
	if (this->frameCount * SPP >= this->targetSPP)
	{
		this->break_flag = 1;
	}
}

void RTRendererCUDAQT::Startear()
{
	looper->start();
}

void RTRendererCUDAQT::Stop()
{
	this->looper->break_flag = 1;
}

void RTRendererCUDAQT::ShowPara()
{
	std::string str = "Current SPP Threshold =" + std::to_string(this->looper->targetSPP);
	this->logText->append(QString::fromStdString(str));
	str = "Current Clip Upperbound =" + std::to_string(this->looper->targetClipUpperbound);
	this->logText->append(QString::fromStdString(str));
}

void RTRendererCUDAQT::setParameter()
{
	this->looper->targetSPP = this->paraTargetSPP->text().toInt();//SPP
	if (this->looper->targetClipUpperbound != this->paraClipUpperbound->text().toInt())
		this->looper->reset_flag = true;
	this->looper->targetClipUpperbound = this->paraClipUpperbound->text().toFloat();
	

	this->ShowPara();
	if (this->looper->reset_flag)
	{
		this->looper->reset_flag = false;
		this->looper->frameCount = 0;
	}
}

void RTRendererCUDAQT::discardParameter()
{
	this->paraTargetSPP->setText(QString::fromStdString(std::to_string(this->looper->targetSPP)));
	this->paraClipUpperbound->setText(QString::fromStdString(std::to_string(this->looper->targetClipUpperbound)));
}