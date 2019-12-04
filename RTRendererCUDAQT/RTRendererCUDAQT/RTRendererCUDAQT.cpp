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
	setGeometry(10, 10, MAX_X + 300, MAX_Y + 300);

	Lab = new QLabel(this);
	Lab->setGeometry(20, 20, MAX_X, MAX_Y);

	StartButton = new QPushButton("Start", this);
	StartButton->setGeometry(20,MAX_Y + 40 ,100 ,50 );

	StopButton = new QPushButton("Stop", this);
	StopButton->setGeometry(140, MAX_Y + 40 , 100, 50);
	
	ParameterText = new QTextEdit(this);
	ParameterText->setGeometry(MAX_X + 50, 20, 200, MAX_Y);

	Updater = new QPushButton("Update", this);
	Updater->setGeometry(260, MAX_Y + 80, 100, 30);

	Para_Stop = new QLineEdit(this);
	Para_Stop->setGeometry(260, MAX_Y + 40, 100, 30);

	looper = new LoopThread(this);
	connect(StartButton, SIGNAL(clicked()), this, SLOT(Startear()));
	connect(looper, SIGNAL(refresh_flag()), this, SLOT(refresh()));
	connect(StopButton, SIGNAL(clicked()), this, SLOT(Stop()));
	connect(Updater, SIGNAL(clicked()), this, SLOT(ShowPara()));
}

void RTRendererCUDAQT::Startear()
{
	looper->start();
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


void RTRendererCUDAQT::Stop()
{
	this->looper->break_flag = 1;
}

void RTRendererCUDAQT::ShowPara()
{
	int pre_b = this->Para_Stop->text().toInt();
	std::string str = "Current Break_Pre =";
	std::string ing;
	std::stringstream ss;
	ss << pre_b;
	ss >> ing;
	str += ing;
	this->looper->pre_break = pre_b;
	this->ParameterText->append(QString::fromStdString(str));
}

